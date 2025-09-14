from torch.autograd import Variable
from .methods_helper import *
from .utils import *

from qwen_vl_utils import process_vision_info


# OpenAI CLIP 归一化常量
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def pil_to_clip_tensor_bcwh(img, requires_grad=True, dtype=torch.float32, device=None):
    """
    将 PIL.Image 转为按 OpenAI CLIP 归一化的 tensor，维度为 [B, C, W, H]（B=1）。
    不进行 resize / crop；仅做 RGB 转换、[0,1] 归一化与标准化。
    
    Args:
        img: PIL.Image 或可被 PIL.Image.open 读取的路径
        requires_grad (bool): 返回的 tensor 是否需要梯度
        dtype: 返回 tensor 的数据类型（默认 float32）
        device: 返回 tensor 的设备（例如 "cuda" 或 torch.device(...)）

    Returns:
        x_bcwh: torch.Tensor, 形状 [1, 3, W, H]
    """
    # 1) 读图并确保 RGB
    if isinstance(img, (str, bytes, np.ndarray)):
        img = Image.open(img)
    img = img.convert("RGB")
    
    w, h = img.size
    new_w = round(w / 28) * 28
    new_h = round(h / 28) * 28
    
    img = img.resize((new_w, new_h), Image.BICUBIC)

    # 2) PIL -> numpy -> torch，归一到 [0,1]
    np_img = np.asarray(img, dtype=np.float32) / 255.0        # [H, W, 3]
    x = torch.from_numpy(np_img)                              # [H, W, 3]
    x = x.to(dtype=dtype)

    # 3) HWC -> CHW
    x = x.permute(2, 0, 1)                                    # [3, H, W]

    # 4) 按 CLIP 均值方差标准化
    mean = torch.tensor(OPENAI_CLIP_MEAN, dtype=dtype).view(3, 1, 1)
    std  = torch.tensor(OPENAI_CLIP_STD,  dtype=dtype).view(3, 1, 1)
    x = (x - mean) / std                                      # [3, H, W]

    # 5) 加 batch 维 -> [1, 3, H, W]
    x = x.unsqueeze(0)                                        # [1, 3, H, W]

    # 6) 设备与梯度设置
    if device is not None:
        x = x.to(device)
    x.requires_grad_(requires_grad)

    return x

def tensor2pack(patches: torch.Tensor) -> torch.Tensor:
    temporal_patch_size=2
    resized_height = patches.shape[2]
    resized_width = patches.shape[3]
    patch_size = 14
    merge_size = 2
    
    # 如果 B 不能整除 temporal_patch_size，就补齐
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = patches[-1].unsqueeze(0).repeat(temporal_patch_size - 1, 1, 1, 1)
        patches = torch.cat([patches, repeats], dim=0)

    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )

    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()

    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    return flatten_patches

def gen_explanations_qwenvl(model, processor, image, text_prompt, tokenizer, positions=None, select_word_id=None):
    """_summary_

    Args:
        model (_type_): _description_
        processor (_type_): _description_
        image (_type_): PIL格式图片
        text_prompt (_type_): _description_
        device (_type_): _description_
    """
    input_size = (image.size[1], image.size[0])
    size=32
    opt = 'NAG'
    diverse_k = 1
    init_posi = 0
    init_val = 0.
    L1 = 1.0
    L2 = 0.1
    gamma = 1.0
    L3 = 10.0
    momentum = 5
    ig_iter = 10
    iterations=5
    lr=10
    
    method = iGOS_pp
    
    i_obj = 0
    total_del, total_ins, total_time = 0, 0, 0
    all_del_scores = []
    all_ins_scores = []
    save_list = []
    
    # 开始处理数据
    image_size = [image.size]
    kernel_size = get_kernel_size(image.size)
    
    blur = cv2.GaussianBlur(np.asarray(image), (kernel_size, kernel_size), sigmaX=kernel_size-1)
    blur = Image.fromarray(blur.astype(np.uint8))
    
    # tensor
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": blur},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
            messages1, tokenize=False, add_generation_prompt=True)
    image_tensor, _ = process_vision_info(messages1)
    blur_tensor, _ = process_vision_info(messages2)
    
    inputs = processor(
            text=[text],
            images=image_tensor,    # 这里可以多个
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    inputs_blur = processor(
            text=[text],
            images=blur_tensor,    # 这里可以多个
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    
    image_tensor = inputs['pixel_values']
    blur_tensor = inputs_blur['pixel_values']
    
    input_ids = inputs['input_ids']
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            do_sample=False,      # Disable sampling and use greedy search instead
            num_beams=1,          # Set to 1 to ensure greedy search instead of beam search.
            max_new_tokens=128)
        generated_ids_trimmed = [   # 去掉图像和prompt的文本
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    selected_token_word_id = generated_ids_trimmed[0].cpu().numpy().tolist()
    selected_token_id = [i for i in range(len(selected_token_word_id))]
    target_token_position = np.array(selected_token_id) + len(inputs['input_ids'][0])
    
    if positions == None:
        positions, keywords = find_keywords(model, inputs, generated_ids, generated_ids_trimmed, image_tensor, blur_tensor, target_token_position, selected_token_word_id, tokenizer)
    else:
        keywords = processor.batch_decode(
            generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[positions[0]]
    
    print(keywords)
    
    if select_word_id != None:
        for position, word_id in zip(positions, select_word_id):
            generated_ids_trimmed[0][position] = word_id
    
    pred_data=dict()
    pred_data['labels'] = generated_ids_trimmed
    pred_data['keywords'] = positions
    pred_data['boxes'] = np.array([[0, 0, input_size[0], input_size[1]]])
    pred_data['no_res'] = False
    pred_data['pred_text'] = output_text
    pred_data['keywords_text'] = keywords
    
    # calculate init area
    pred_data = get_initial(pred_data, k=diverse_k, init_posi=init_posi, 
                           init_val=init_val, input_size=input_size, out_size=size)
    for l_i, label in enumerate(pred_data['labels']):
        label = label.unsqueeze(0)
        keyword = pred_data['keywords']
        now = time.time()
        masks, loss_del, loss_ins, loss_l1, loss_tv, loss_l2, loss_comb_del, loss_comb_ins = method(
                model=model,
                inputs = inputs, 
                generated_ids=generated_ids,
                init_mask=pred_data['init_masks'][0],
                image=pil_to_clip_tensor_bcwh(image).to(model.device),
                target_token_position=target_token_position, selected_token_word_id=selected_token_word_id,
                baseline=pil_to_clip_tensor_bcwh(blur).to(model.device),
                label=label,
                size=size,
                iterations=iterations,
                ig_iter=ig_iter,
                L1=L1,
                L2=L2,
                L3=L3,
                lr=lr,
                opt=opt,
                prompt=input_ids,
                image_size=image_size,
                positions=keyword,
                resolution=None,
                processor=tensor2pack
            )
        total_time += time.time() - now
        
        masks = masks[0,0].detach().cpu().numpy()
        masks -= np.min(masks)
        masks /= np.max(masks)
        
        image = np.array(image)
        masks = cv2.resize(masks, (image.shape[1], image.shape[0]))
        
        heatmap = np.uint8(255 * (1-masks))  
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_image = image
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        # cv2.imwrite("igos++.jpg", superimposed_img)
        
    return masks, superimposed_img

def gen_explanations_internvl(model, processor, image, text_prompt, tokenizer, positions=None):
    input_size = (image.size[1], image.size[0])
    size=32
    opt = 'NAG'
    diverse_k = 1
    init_posi = 0
    init_val = 0.
    L1 = 1.0
    L2 = 0.1
    gamma = 1.0
    L3 = 10.0
    momentum = 5
    ig_iter = 10
    iterations=5
    lr=10
    
    method = iGOS_pp
    
    i_obj = 0
    total_del, total_ins, total_time = 0, 0, 0
    all_del_scores = []
    all_ins_scores = []
    save_list = []
    
    # 开始处理数据
    image_size = [image.size]
    kernel_size = get_kernel_size(image.size)
    
    # tensor
    messages1 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    
    # Preparation for inference
    inputs = processor.apply_chat_template(messages1, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    # inputs_blur = processor.apply_chat_template(messages1, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

    image_tensor = inputs['pixel_values']
    # blur_tensor = inputs_blur['pixel_values']
    blur_tensor = image_tensor * 0  # blur image cant choose salient word
    
    input_ids = inputs['input_ids']
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            do_sample=False,      # Disable sampling and use greedy search instead
            num_beams=1,          # Set to 1 to ensure greedy search instead of beam search.
            max_new_tokens=128)
        generated_ids_trimmed = [   # 去掉图像和prompt的文本
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    selected_token_word_id = generated_ids_trimmed[0].cpu().numpy().tolist()
    selected_token_id = [i for i in range(len(selected_token_word_id))]
    target_token_position = np.array(selected_token_id) + len(inputs['input_ids'][0])
    
    if positions == None:
        positions, keywords = find_keywords(model, inputs, generated_ids, generated_ids_trimmed, image_tensor, blur_tensor, target_token_position, selected_token_word_id, tokenizer)
    else:
        keywords = processor.batch_decode(
            generated_ids_trimmed[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[positions[0]]
    
    
    print(keywords)
    
    pred_data=dict()
    pred_data['labels'] = generated_ids_trimmed
    pred_data['keywords'] = positions
    pred_data['boxes'] = np.array([[0, 0, input_size[0], input_size[1]]])
    pred_data['no_res'] = False
    pred_data['pred_text'] = output_text
    pred_data['keywords_text'] = keywords
    
    
    # new image
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((448, 448))},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(messages2, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    
    y = torch.stack(generated_ids_trimmed, dim=0)


    generated_ids = torch.cat([inputs["input_ids"], y if y.dim()==2 else y.unsqueeze(0)], dim=1).to(model.device)
    # inputs['attention_mask'] = torch.ones_like(inputs["input_ids"]).to(model.device)
    
    
    target_token_position = np.array(selected_token_id) + len(inputs['input_ids'][0])
    # calculate init area
    pred_data = get_initial(pred_data, k=diverse_k, init_posi=init_posi, 
                           init_val=init_val, input_size=input_size, out_size=size)
    for l_i, label in enumerate(pred_data['labels']):
        label = label.unsqueeze(0)
        keyword = pred_data['keywords']
        now = time.time()
        masks, loss_del, loss_ins, loss_l1, loss_tv, loss_l2, loss_comb_del, loss_comb_ins = method(
                model=model,
                inputs = inputs, 
                generated_ids=generated_ids,
                init_mask=pred_data['init_masks'][0],
                image=inputs['pixel_values'][-1].unsqueeze(0).to(model.device),
                target_token_position=target_token_position, selected_token_word_id=selected_token_word_id,
                baseline=inputs['pixel_values'][-1].unsqueeze(0).to(model.device)*0,
                label=label,
                size=size,
                iterations=iterations,
                ig_iter=ig_iter,
                L1=L1,
                L2=L2,
                L3=L3,
                lr=lr,
                opt=opt,
                prompt=input_ids,
                image_size=image_size,
                positions=keyword,
                resolution=None,
                processor=None
            )
        total_time += time.time() - now
        
        masks = masks[0,0].detach().cpu().numpy()
        masks -= np.min(masks)
        masks /= np.max(masks)
        
        image = np.array(image)
        masks = cv2.resize(masks, (image.shape[1], image.shape[0]))
        
        heatmap = np.uint8(255 * (1-masks))  
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_image = image
        superimposed_img = heatmap * 0.4 + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        # cv2.imwrite("igos++.jpg", superimposed_img)
        
    return masks, superimposed_img

def interval_score(model, inputs, generated_ids, images, target_token_position, selected_token_word_id, baseline, up_masks, num_iter, noise=True, positions=None, processor=None):
    # if model_name == 'llava' or model_name == 'llava_next' or model_name == 'mgm':
    # The intervals to approximate the integral over
    intervals = torch.linspace(1/num_iter, 1, num_iter, requires_grad=False).cuda().view(-1, 1, 1, 1)
    interval_masks = up_masks.unsqueeze(1).to(model.device) * intervals.to(model.device)
    local_images = phi(images.unsqueeze(1), baseline.unsqueeze(1), interval_masks)

    if noise:
        local_images = local_images + torch.randn_like(local_images) * .2

    local_images = local_images.transpose(0, 1)
    # input_ids = torch.cat((prompt, label), dim=1)
    positions = torch.tensor(positions).to(model.device)

    losses = torch.tensor(0.).to(model.device)
    for single_img in local_images:
        # single_img = single_img.half()
        
        if processor == None:
            single_input = single_img
        else:
            single_input = processor(single_img)
        
        probs = pred_probs(model, inputs, generated_ids, single_input, target_token_position, selected_token_word_id, need_grad=True)
        #losses += probs[positions].mean()
        losses += torch.log(probs)[positions].sum()

    return losses / num_iter


def integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks, num_iter, noise=True,  positions=None,processor=None):
    loss = interval_score(
        model, 
        inputs, 
        generated_ids, 
        image, 
        target_token_position, 
        selected_token_word_id,
        baseline, 
        up_masks, 
        num_iter, 
        noise=noise, 
        positions=positions,
        processor=processor)
    
    loss.sum().backward(retain_graph=True)
    return loss.sum().item()

def iGOS_pp(
        model,
        inputs, 
        generated_ids,
        init_mask,
        image,
        target_token_position, 
        selected_token_word_id,
        baseline,
        label,
        size=32,
        iterations=15,
        ig_iter=20,
        L1=1,
        L2=1,
        L3=20,
        lr=1000,
        opt='LS',
        softmax=True,
        processor=None,
        **kwargs):

    L2 = 0.1
    gamma = 1.0
    momentum = 5
    
    def regularization_loss(image, masks):
        return L1 * torch.mean(torch.abs(1 - masks).view(masks.shape[0], -1), dim=1), \
               L3 * bilateral_tv_norm(image, masks, tv_beta=2, sigma=0.01), \
               L2 * torch.sum((1 - masks)**2, dim=[1, 2, 3])

    def ins_loss_function(up_masks, indices, noise=True):
        losses = -interval_score(
            model, 
            inputs, 
            generated_ids, 
            baseline[indices], 
            target_token_position, 
            selected_token_word_id, 
            image[indices], 
            up_masks, 
            ig_iter, 
            noise, 
            positions)
        
        return losses.sum(dim=1).view(-1)

    def del_loss_function(up_masks, indices, noise=True):
        losses = interval_score(
            model, 
            inputs, 
            generated_ids, 
            baseline[indices], 
            target_token_position, 
            selected_token_word_id, 
            image[indices], 
            up_masks, 
            ig_iter, 
            noise, 
            positions)
        return losses.sum(dim=1).view(-1)

    def loss_function(up_masks, masks, indices):
        loss = del_loss_function(up_masks[:, 0], indices)
        loss += ins_loss_function(up_masks[:, 1], indices)
        loss += del_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        loss += ins_loss_function(up_masks[:, 0] * up_masks[:, 1], indices)
        return loss + regularization_loss(image[indices], masks[:, 0] * masks[:, 1])

    masks_del = torch.ones((1, 1, size, size), dtype=torch.float32, device='cuda')
    masks_del = masks_del * init_mask.cuda()
    masks_del = Variable(masks_del, requires_grad=True)
    masks_ins = torch.ones((image.shape[0], 1, size, size), dtype=torch.float32, device='cuda')
    masks_ins = masks_ins * init_mask.cuda()
    masks_ins = Variable(masks_ins, requires_grad=True)
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)
    resolution = kwargs.get('resolution', None)

    
    if opt == 'NAG':
        cita_d=torch.zeros(1).cuda()
        cita_i=torch.zeros(1).cuda()
    
    prompt = kwargs.get('prompt', None)
    image_size = kwargs.get('image_size', None)
    positions = kwargs.get('positions', None)
    losses_del, losses_ins, losses_l1, losses_tv, losses_l2, losses_comb_del, losses_comb_ins = [], [], [], [], [], [], []
    for i in range(iterations):
        up_masks1 = upscale(masks_del, image)
        up_masks2 = upscale(masks_ins, image)

        # Compute the integrated gradient for the combined mask, optimized for deletion
        loss_comb_del = integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks1 * up_masks2, ig_iter, positions=positions, processor=processor)
        
        total_grads1 = masks_del.grad.clone()
        total_grads2 = masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the combined mask, optimized for insertion
        loss_comb_ins = integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks1 * up_masks2, ig_iter, positions=positions, processor=processor)
        
        total_grads1 -= masks_del.grad.clone()  # Negative because insertion loss is 1 - score.
        total_grads2 -= masks_ins.grad.clone()
        masks_del.grad.zero_()
        masks_ins.grad.zero_()

        # Compute the integrated gradient for the deletion mask
        loss_del = integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks1, ig_iter, positions=positions, processor=processor)
        
        total_grads1 += masks_del.grad.clone()
        masks_del.grad.zero_()

        # Compute the integrated graident for the insertion mask
        loss_ins = integrated_gradient(model, inputs, generated_ids, image, target_token_position, selected_token_word_id, baseline, up_masks2, ig_iter, positions=positions, processor=processor)
        
        total_grads2 -= masks_ins.grad.clone()
        masks_ins.grad.zero_()

        # Average them to balance out the terms with the regularization terms
        total_grads1 /= 2
        total_grads2 /= 2

        # Computer regularization for combined masks
        L2 = exp_decay(L2, i, gamma)
        loss_l1, loss_tv, loss_l2 = regularization_loss(image, masks_del * masks_ins)
        losses = loss_l1 + loss_tv + loss_l2
        losses.sum().backward()
        total_grads1 += masks_del.grad.clone()
        total_grads2 += masks_ins.grad.clone()

        if opt == 'LS':
            masks = torch.cat((masks_del.unsqueeze(1), masks_ins.unsqueeze(1)), 1)
            total_grads = torch.cat((total_grads1.unsqueeze(1), total_grads2.unsqueeze(1)), 1)
            lrs = line_search(masks, total_grads, loss_function, lr)
            masks_del.data -= total_grads1 * lrs
            masks_ins.data -= total_grads2 * lrs
        
        if opt == 'NAG':
            e = i / (i + momentum)
            cita_d_p = cita_d
            cita_i_p = cita_i
            cita_d = masks_del.data - lr * total_grads1
            cita_i = masks_ins.data - lr * total_grads2
            masks_del.data = cita_d + e * (cita_d - cita_d_p)
            masks_ins.data = cita_i + e * (cita_i - cita_i_p)

        masks_del.grad.zero_()
        masks_ins.grad.zero_()
        masks_del.data.clamp_(0,1)
        masks_ins.data.clamp_(0,1)

        losses_del.append(loss_del)
        losses_ins.append(loss_ins)
        losses_comb_del.append(loss_comb_del)
        losses_comb_ins.append(loss_comb_ins)
        losses_l1.append(loss_l1.item())
        losses_tv.append(loss_tv.item())
        losses_l2.append(loss_l2.item())
        print(f'iteration: {i} lr: {lr:.4f} loss_comb_del: {loss_comb_del:.4f}, loss_comb_ins: {loss_comb_ins:.4f}, loss_del: {loss_del:.4f}, loss_ins: {loss_ins:.4f}, loss_l1: {loss_l1.item():.4f}, loss_tv: {loss_tv.item():.4f}, loss_l2: {loss_l2.item():.4f}')

    return masks_del * masks_ins, losses_del, losses_ins, losses_l1, losses_tv, losses_l2, losses_comb_del, losses_comb_ins


