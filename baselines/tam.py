import os, torch, cv2, subprocess
import fitz
import numpy as np
from scipy.optimize import minimize_scalar
from pathlib import Path


def rank_guassian_filter(img, kernel_size=3):
    """
    Apply a rank-based Gaussian-weighted filter for robust activation map denoising.

    Parameters:
    img : np.ndarray
        Input 2D grayscale image.
    kernel_size : int
        Size of the square kernel (must be odd).

    Returns:
    filtered_img : np.ndarray
        Denoised image after applying the Gaussian weighted rank filter.

    Note:
        The sigma (std) of is refined to coefficient of variation for robust results
    """

    filtered_img = np.zeros_like(img)
    pad_width = kernel_size // 2
    padded_img = np.pad(img, pad_width, mode='reflect')
    ax = np.array(range(kernel_size ** 2)) - kernel_size ** 2 // 2

    for i in range(pad_width, img.shape[0] + pad_width):
        for j in range(pad_width, img.shape[1] + pad_width):
            window = padded_img[i - pad_width:i + pad_width + 1,
                                j - pad_width:j + pad_width + 1]

            sorted_window = np.sort(window.flatten())
            mean = sorted_window.mean()
            if mean > 0:
                sigma = sorted_window.std() / mean # std -> cov
                kernel = np.exp(-(ax**2) / (2 * sigma**2))
                kernel = kernel / np.sum(kernel)
                value = (sorted_window * kernel).sum()
            else:
                value = 0
            filtered_img[i - pad_width, j - pad_width] = value
    
    return filtered_img


def least_squares(map1, map2):
    """
    Find the scalar that minimizes the squared difference between map1 and scalar * map2.

    Args:
        map1 (np.ndarray): First data array.
        map2 (np.ndarray): Second data array.

    Returns:
        float: Optimal scalar multiplier.
    """

    def diff(x, map1, map2):
        return np.sum((map1 - map2 * x)**2)

    result = minimize_scalar(diff, args=(map1, map2))
    return result.x


def generate_latex(words, relevances, cmap="bwr", font=r'{18pt}{21pt}'):
    """
    Generate LaTeX code to visualize tokens with colored backgrounds or text, based on their relevance scores.

    Args:
        words (list of str): List of token strings, where tokens starting with '▁' or 'Ġ' represent spaces.
        relevances (list of float): List of relevance scores corresponding to each token.
            - relevance >= 0: earlier context tokens, color-coded with a jet colormap.
            - relevance == -1: current explained token, shown with black background and white text.
            - relevance == -2: next tokens, rendered in gray color.
            - relevance == -3: special marker to add a newline and "Candidates:" label.
            - relevance == -4: special marker to add a newline and print the word string as is.
        cmap (str): Colormap to use for positive relevances (default "bwr" - unused in current code).
        font (str): Font size and line spacing in LaTeX format, e.g. '{18pt}{21pt}'.

    Returns:
        str: A complete LaTeX document as a string with colored tokens visualized.
    """


    latex_code = r'''
    \documentclass[arwidth=200mm]{standalone}
    \renewcommand{\normalsize}{\fontsize''' + font + r'''\selectfont}
    \usepackage[dvipsnames]{xcolor}

    \begin{document}
    \fbox{
    \parbox{\textwidth}{
    \setlength\fboxsep{0pt}
    '''

    for i in range(len(words)):
        word = words[i]
        relevance = relevances[i]

        # relevance >= 0 for earlier context tokens (jet colors)
        if relevance >= 0:
            jet_colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
            b, g, r = jet_colormap[int(relevances[i] * 255)][0].tolist()
            if word[:2] == '$ ' and word[-1] == '$': # candidates
                latex_code += f' \\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}, '
            elif word.startswith('▁') or word.startswith('Ġ') or word.startswith(' '):
                word = word.replace('▁', ' ').replace('Ġ', ' ')
                latex_code += f' \\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'
            else:
                latex_code += f'\\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'

        # for current explained token (black)
        elif relevance == -1:
            if word.startswith('▁') or word.startswith('Ġ') or word.startswith(' '):
                word = word.replace('▁', ' ').replace('Ġ', ' ')
                latex_code += f' \\textbf{{\\colorbox[RGB]{{{0},{0},{0}}}{{\\textcolor[RGB]{{{255},{255},{255}}}{{\\strut {word}}}}}}}'
            else:
                latex_code += f'\\textbf{{\\colorbox[RGB]{{{0},{0},{0}}}{{\\textcolor[RGB]{{{255},{255},{255}}}{{\\strut {word}}}}}}}'

        # for next tokens (gray)
        elif relevance == -2:
            b, g, r = 200, 200, 200
            if word.startswith('▁') or word.startswith('Ġ') or word.startswith(' '):
                word = word.replace('▁', ' ').replace('Ġ', ' ')
                latex_code += f' \\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'
            else:
                latex_code += f'\\textbf{{\\textcolor[RGB]{{{r},{g},{b}}}{{\\strut {word}}}}}'

        # for top pred
        elif relevance == -3:
            latex_code += '\\\\$Candidates:$'

        # for custom vis str
        elif relevance == -4:
            latex_code += '\\\\' + word

    latex_code += r'}}\end{document}'

    return latex_code


def compile_latex_to_jpg(latex_code, path='word_colors.pdf', delete_aux_files=True, dpi=500):
    """
    Compile a LaTeX string into a JPG image.

    Parameters:
    - latex_code (str): The LaTeX source code to compile.
    - path (str or Path): File path for intermediate PDF and auxiliary files. The output image is returned as an array.
    - delete_aux_files (bool): Whether to delete auxiliary files (.aux, .log, .tex, .pdf) after compilation.
    - dpi (int): Resolution for the output image in dots per inch.

    Returns:
    - img (numpy.ndarray): The compiled LaTeX rendered as a color image (BGR) array.
                          Returns None if compilation fails.
    """

    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    with open(path.with_suffix(".tex"), 'w') as f:
        f.write(latex_code)

    try:
        res_code = subprocess.run(['xelatex', '--output-directory', path.parent, path.with_suffix(".tex")], \
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except:
        print('Skip, fail to compile: ' + res_code)
        return None

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    page = fitz.open(path.with_suffix(".pdf")).load_page(0)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    if delete_aux_files:
        for suffix in ['.aux', '.log', '.tex', '.pdf']:
            os.remove(path.with_suffix(suffix))

    getpngdata = pix.tobytes("png")
    image_array = np.frombuffer(getpngdata, dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)[:,:,:3]
    return img


def vis_text(words, relevances, candidates, candi_scores, vis_token_idx, path='heatmap.jpg', font=r'{18pt}{21pt}'):
    """
    Visualizes text tokens and their relevance scores as a heatmap image using LaTeX.

    This function processes a list of words and their corresponding relevance scores, along with candidate tokens 
    and their scores, to create a color-coded heatmap visualization. It handles special LaTeX characters by escaping 
    them appropriately to ensure correct LaTeX rendering. The visualization includes the explained tokens, subsequent 
    tokens, and top prediction candidates with distinct coloring based on their scores.

    Args:
        words: All tokens need to visualize.
        relevances: Relevance scores corresponding to each token.
        candidates: Candidate tokens (top k predictions).
        candi_scores: Scores associated with each candidate token.
        vis_token_idx (int): Index of the token to vis (explain).
        path (str, optional): File path to save the generated heatmap image. Defaults to 'heatmap.jpg'.
        font (str, optional): LaTeX font size settings for the visualization. Defaults to r'{18pt}{21pt}'.

    Returns:
        str: Numpy image for the visualized texts
    """


    # add scores (-2, gray) for next tokens after the exaplained one
    add_scores = []
    for i in range(len(relevances), len(words[:-1])):
        add_scores.append(-2)

    # explained tokens + next tokens + top pred candidates (see defination of scores in generate_latex)
    all_scores = relevances.tolist() + add_scores + [-3] + candi_scores.cpu().float().tolist()
    all_scores[vis_token_idx] = -1

    # scores correspond to the words
    all_words = words[:-1] + [''] + ['$ ' + _ + '$' for _ in candidates]

    # replace special texts to fit latex
    all_words = [_.replace('\\', '\\backslash').replace('\n', '\\newline').replace('_', '\\_').replace('^', '\\^').replace('&', '\\&').replace('%', '\\%').replace('Ċ', '\\newline') for _ in all_words]

    # to latex, then to img
    latex_code = generate_latex(all_words, all_scores, cmap='bwr', font=font)
    return compile_latex_to_jpg(latex_code, path=path, delete_aux_files=True)


def multimodal_process(raw_img, vision_shape, img_scores, txt_scores, txts, candidates, candi_scores, \
                       vis_token_idx, img_save_fn, eval_only=False, vis_width=-1):
    """
    Process multimodal tokens: visualizing combined image and text activations with normalizing, filtering, and blending scores.

    This function processes image and text token scores to generate a multimodal visualization:
    - Normalizes image and text token scores together for comparability.
    - Applies the Rank Gank Guassian Filter for vision tokens.
    - Visualizes text token via latex.
    - Combines visual maps of image and text tokens for final output.
    - Supports single image, multiple images, and video batch inputs.
    - Optionally returns only evaluation maps without visualization.

    Args:
        raw_img (np.ndarray or list of np.ndarray): Raw input image(s). For multiple images, provide a list.
        vision_shape (tuple or list of tuples): Shape(s) of vision tokens (height, width) or batch size + shape for video.
        img_scores (np.ndarray): Activation scores for image tokens.
        txt_scores (np.ndarray): Activation scores for text tokens.
        txts (list): Visualized texts, including texts before the target and next words.
        candidates (list): Candidate topK predictions of the explianed token.
        candi_scores (np.ndarray): Scores for candidate tokens.
        vis_token_idx (list): Index of the explained token in all_text to visualize.
        img_save_fn (str): Path to save the visualization image.
        eval_only (bool, optional): If True, only returns evaluation score maps without visualization. Defaults to False.
        vis_width (int, optional): Width for resizing images and visualizations. If -1, no resizing is done. Defaults to -1.

    Returns:
        tuple:
            - out_img (np.ndarray or None): Final blended visualization image combining image and text scores.
            - img_map (np.ndarray or list of np.ndarray): Evaluation score maps for image tokens.
    """


    # normalize multimodal tokens
    txt_scores = txt_scores[:-1] # ignore self score
    all_scores = np.concatenate([img_scores, txt_scores], 0)
    all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min())
    img_scores = all_scores[:len(img_scores)]
    txt_scores = all_scores[len(img_scores):]

    eval_only = True if img_save_fn == "" else False

    # for multiple imgs
    if isinstance(vision_shape[0], tuple):
        resized_img, img_map = [], []
        start_idx = 0
        for n in range(len(vision_shape)):
            t_h, t_w = vision_shape[n]
            h, w, c = raw_img[n].shape

            # for fix height
            if vis_width > 0:
                h = int(vis_width)
                w = int(float(w) / h * vis_width)

            # apply the rank_guassian_filter for vision tokens of each img
            end_idx = start_idx + int(t_h * t_w)
            img_map_ = rank_guassian_filter(img_scores[start_idx: end_idx].reshape(t_h, t_w), 3)
            start_idx = end_idx
            img_map_ = (img_map_ * 255).astype('uint8')

            # resize map and raw img if need vis
            if not eval_only:
                img_map_ = cv2.applyColorMap(img_map_, cv2.COLORMAP_JET)
                img_map_ = cv2.resize(img_map_, (w, h))
                if vis_width > 0:
                    raw_img_ = cv2.resize(raw_img[n], (w, h))
                    resized_img.append(raw_img_)

            img_map.append(img_map_)

        # eval only output
        if eval_only:
            return None, img_map

        out_img = [img_map[i] * 0.5 + resized_img[i] * 0.5 for i in range(len(vision_shape))]
        out_img = np.concatenate(out_img, 1)

        # text vis via latex
        try:
            txt_map = vis_text(txts, txt_scores, candidates, candi_scores, vis_token_idx, path=img_save_fn, font=r'{5pt}{6pt}')
        except:
            print('Skip text visualization, please check the installation of texlive-xetex.')
            return out_img, img_map
        
        if not isinstance(txt_map, np.ndarray):
            print('Skip txt visualization, please check weather the text special character compatible with LaTeX.')
            return out_img, img_map

        # concat multimodal vis
        txt_map = cv2.resize(txt_map, (out_img.shape[1], int(float(txt_map.shape[0]) / float(txt_map.shape[1]) * out_img.shape[1])))
        out_img = np.concatenate([out_img, txt_map], 0)

        return out_img, img_map

    # single img
    elif len(vision_shape) == 2:
        # set img size
        t_h, t_w = vision_shape
        h, w, c = raw_img.shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        # apply filter
        img_scores = rank_guassian_filter(img_scores.reshape(t_h, t_w), 3)
        img_scores = (img_scores * 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_map = cv2.applyColorMap(img_scores, cv2.COLORMAP_JET)
        img_map = cv2.resize(img_map, (w, h))
        if vis_width > 0:
            raw_img = cv2.resize(raw_img, (w, h))
        out_img = img_map * 0.5 + raw_img * 0.5

        # vis text via latex
        try:
            txt_map = vis_text(txts, txt_scores, candidates, candi_scores, vis_token_idx, path=img_save_fn)
        except:
            print('Skip text visualization, please check the installation of texlive-xetex.')
            return out_img, img_scores

        if not isinstance(txt_map, np.ndarray):
            print('Skip txt visualization, please check weather the text special character compatible with LaTeX.')
            return out_img, img_scores

        txt_map = cv2.resize(txt_map, (w, int(float(txt_map.shape[0]) / float(txt_map.shape[1]) * w)))
        out_img = np.concatenate([out_img, txt_map], 0)

        return out_img, img_scores

    # video
    else:
        b, t_h, t_w = vision_shape
        h, w, c = raw_img[0].shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        img_scores = np.array([rank_guassian_filter(_.reshape(t_h, t_w), 3) for _ in np.array_split(img_scores, b)])
        img_scores = (img_scores * 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_map = [cv2.resize(cv2.applyColorMap(_, cv2.COLORMAP_JET), (w, h)) for _ in img_scores]
        if vis_width > 0:
            raw_img = [cv2.resize(_, (w, h)) for _ in raw_img]
        out_img = [img_map[i] * 0.5 + raw_img[i] * 0.5 for i in range(b)]
        out_img = np.concatenate(out_img, 1)

        # vis text via latex
        try:
            txt_map = vis_text(txts, txt_scores, candidates, candi_scores, vis_token_idx, path=img_save_fn, font=r'{5pt}{6pt}')
        except:
            print('Skip text visualization, please check the installation of texlive-xetex.')
            return out_img, img_scores

        if not isinstance(txt_map, np.ndarray):
            print('Skip txt visualization, please check weather the text special character compatible with LaTeX.')
            return out_img, img_scores

        txt_map = cv2.resize(txt_map, (int(w * b), int(float(txt_map.shape[0]) / float(txt_map.shape[1]) * w * b)))
        out_img = np.concatenate([out_img, txt_map], 0)

        return out_img, img_scores



def id2idx(inp_id, target_id, return_last=False):
    """
    Convert a target ID or sequence of IDs to the corresponding index in the input list.

    Args:
        input_ids (list of int): The list of token IDs to search within.
        target_id (int or list of int): The target token ID or sequence of token IDs to find.
        return_last (bool): If True and target_id is a list, return the index of the last token in the matched sequence.
                            Otherwise, return the index of the first token.

    Returns:
        int: The index of the target ID (or start/end of the sequence) in input_ids, or -1 if not found.
    """

    # use a array of tokens as the identifier
    if isinstance(target_id, list):
        n = len(target_id)
        indexes = [i for i in range(len(inp_id) - n + 1) if inp_id[i:i+n] == target_id]
        if len(indexes) > 0:
            # get the idx of the first token as the end identifier
            idx = indexes[-1]

            # get the idx of the last token as the begain identifier
            if return_last:
                idx += len(target_id) - 1
        else:
            idx = -1

    # if the id is unique, use a int is simple
    else:
        try:
            idx = inp_id.index(target_id)
        except:
            idx = -1
    return idx



def TAM(tokens, vision_shape, logit_list, special_ids, vision_input, \
        processor, save_fn, target_token, img_scores_list, eval_only=False):

    """
    Generate a Token Activation Map (TAM) with optional Estimated Causal Inference (ECI) 
    and Rank Guassian Filter for high quality MLLM visual explaination.

    Args:
        tokens (list): The token sequence including input and generated tokens.
        vision_shape (tuple or list): Shape information of the vision input (image/video).
        logit_list (list of torch.Tensor): List of logits tensors for each generation round; 
        special_ids (dict): Dictionary containing special token ids:
            - 'img_id': list of ids to locate the start and end of vision inputs.
              Note: a int value for img_id indicates all tokens of this id.
            - 'prompt_id': tuple of (start_id, end_id) for prompt text tokens.
            - 'answer_id': tuple of (start_id, end_id) for answer tokens.
            Note: 1. The format is [int/list for start, int/list for end].
                  2. The select tokens are [start + 1: end].
                  3. The start list uses the idx of last token, while end uses the first.
        vision_input (array or list): Raw vision input (images or video frames).
        processor: The model processor to convert tokens to text.
        save_fn (str): File path to save the visualization image (optional).
        target_token (int or tuple): The token index or (round_idx, prompt_token_idx) to explain.
        img_scores_list (list): List to accumulate image maps used in Estimated Causal Inference.
            Note: need to define a empty list for the first round of each example.
        eval_only (bool): Whether to run in evaluation mode (affects visualization size).

    Returns:
        img_map (np.ndarray): The TAM for eval.

    Workflow:
    1. Convert tokens to list and identify indices for image, prompt, and answer tokens.
    2. Decode prompt and answer tokens into text tokens using the processor.
    3. Determine the target token indices and generation round.
    4. For round 0, recursively process all prompt tokens to generate maps.
    5. Extract the logits for the target token's predicted class and compute relevance scores 
       over prompt, answer, and image tokens.
    6. Use Estimated Causal Inference (ECI) with least squares to reduce interference 
       from repeated tokens in the textual input.
    7. Prepare vision input images or frames for visualization.
    8. Identify top candidate tokens to provide context in visualization.
    9. Call multimodal_process to generate the visual explanation map (TAM).
       This step includes the Rank Guassian Filter.
    10. Save the resulting visualization image if a save path is provided.
    11. Return the computed image activation map.

    """

    # start and end id for img, prompt and answer
    img_id = special_ids['img_id']
    prompt_id = special_ids['prompt_id'] # prompt text, start and end id
    answer_id = special_ids['answer_id'] # number of tokens between prompt and answer
    
    # if img_id is a int, take all tokens same to this id
    if len(img_id) == 1:
        img_idx = (np.array(tokens) == img_id[0]).nonzero()[0]
    else:
        img_idx = [id2idx(tokens, img_id[0], True), id2idx(tokens, img_id[1])]

    # convert vocab id to idx in tokens
    prompt_idx = [id2idx(tokens, prompt_id[0], True), id2idx(tokens, prompt_id[1])]
    answer_idx = [id2idx(tokens, answer_id[0], True), id2idx(tokens, answer_id[1])]

    # decode ids

    prompt = processor.tokenizer.tokenize(processor.batch_decode([tokens[prompt_idx[0] + 1: prompt_idx[1]]], \
            skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])
    answer = processor.tokenizer.tokenize(processor.batch_decode([tokens[answer_idx[0] + 1:]], \
            skip_special_tokens=False, clean_up_tokenization_spaces=False)[0])
    txt_all = prompt + answer

    # round_idx indicates the round of generation, this_token_idx is for the exaplained target token
    round_idx = -1
    this_token_idx = 0

    # for non-first rounds
    if isinstance(target_token, int):
        round_idx = target_token
        this_token_idx = -1 # last token of each answer round
        vis_token_idx = len(prompt) + target_token

    # for the first round, which contrains multiple prompt tokens to explain
    else:
        round_idx, prompt_token_idx = target_token
        this_token_idx = prompt_idx[0] + prompt_token_idx + 1
        vis_token_idx = prompt_token_idx

    # vis prompt tokens at round 0
    if round_idx == 0 and isinstance(target_token, int):
        for t in range(len(prompt) + 1):
            # recursion to process prompt tokens
            img_map = TAM(tokens, vision_shape, logit_list, special_ids, vision_input, processor, \
                          save_fn if t == len(prompt) else '', [0, t], img_scores_list, eval_only)

            ## the first prompt token is used to reflect the differenec of activation degrees
            if t == 0:
                first_ori = img_map

        return first_ori

    # assign class id
    if round_idx == 0:

        # last token of round 0 is the first generated token
        if prompt_token_idx == len(prompt):
            this_token_idx = logit_list[0].shape[1] - 1
            cls_id = tokens[this_token_idx]

        # record the first prompt with greedy search
        elif prompt_token_idx == 0:
            cls_id = logit_list[0][0, prompt_idx[0] + 1].argmax(0)

        # other maps prompt tokens
        else:
            cls_id = tokens[this_token_idx]

    # generated tokens (round >= 1)
    else:
        cls_id = tokens[answer_idx[0] + round_idx + 1]

    # class activation map from logits of the target token class
    scores = torch.cat([logit_list[_][0, :, cls_id] for _ in range(round_idx + 1)], -1).clip(min=0)

    # get relevance scores
    scores = scores.detach().cpu().float().numpy()
    prompt_scores = scores[prompt_idx[0] + 1: prompt_idx[1]]
    last_prompt = scores[logit_list[0].shape[1] - 1: logit_list[0].shape[1]]
    answer_scores = scores[answer_idx[0] + 1:]
    txt_scores = np.concatenate([prompt_scores, last_prompt, answer_scores], -1)
    if isinstance(img_idx, list):
        img_scores = scores[img_idx[0] + 1: img_idx[1]]
    else:
        img_scores = scores[img_idx]

    # save img_scores for next Estimated Causal Inference
    img_scores_list.append(img_scores)

    # exclude the same words in ECI
    if len(img_scores_list) > 1 and vis_token_idx < len(txt_all):
        non_repeat_idx = []
        for i in range(vis_token_idx):
            if i < len(txt_all) and txt_all[i] != txt_all[vis_token_idx]:
                non_repeat_idx.append(i)
        txt_scores_ = txt_scores[non_repeat_idx]
        img_scores_list_ = [img_scores_list[_] for _ in non_repeat_idx]

        # get the interference map of ECI
        w = txt_scores_
        w = w / (w.sum() + 1e-8)
        interf_img_scores = (np.stack(img_scores_list_, 0) * w.reshape(-1, 1)).sum(0)

        # apply ECI with the least squares method and relu
        scaled_map = least_squares(img_scores, interf_img_scores)
        img_scores = (img_scores - interf_img_scores * scaled_map).clip(min=0)

    # prepare raw vision input
    if isinstance(vision_shape[0], tuple):
        cv_img = [cv2.cvtColor(np.array(_), cv2.COLOR_RGB2BGR) for _ in vision_input]
    elif len(vision_shape) == 2:
        cv_img = np.array([np.array(vision_input_) for vision_input_ in vision_input])
        # cv_img = np.array(vision_input)
        if len(cv_img.shape) == 4 and cv_img.shape[0] == 1:
            cv_img = cv_img[0]
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    else: #video
        cv_img = [cv2.cvtColor(np.array(_), cv2.COLOR_RGB2BGR) for _ in vision_input[0]]

    # prepare top candidates
    candi_scores, candi_ids = logit_list[round_idx][0, this_token_idx].topk(3)
    candi_scores = candi_scores.softmax(0)
    candidates = processor.batch_decode([[_] for _ in candi_ids])
    
    # apply the multimodal_process to obtain TAM
    vis_img, img_map = multimodal_process(cv_img, vision_shape, img_scores, txt_scores, txt_all, candidates, candi_scores, vis_token_idx, \
            save_fn, eval_only=eval_only, vis_width=-1 if eval_only else 500)
    
    if save_fn != '' and vis_token_idx < (len(txt_all) - 1) and isinstance(vis_img, np.ndarray):
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        cv2.imwrite(save_fn, vis_img)
    
    return img_map