import cv2
import json
import numpy as np
import textwrap

from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable


from sklearn import metrics

matplotlib.get_cachedir()
# plt.rc('font', family="Arial")

def add_value(S_set, json_file):
    single_mask = np.zeros_like(S_set[0])
    single_mask = single_mask.astype(np.float16)
    
    value_list_1 = np.array(json_file["smdl_score"])
    
    # value_list_2 = np.array(
    #     [1 - np.mean(json_file["org_score"]) + np.mean(json_file["baseline_score"])] + json_file["smdl_score"][:-1]
    # )
    value_list_2 = np.array(
        [np.mean(1 - np.array(json_file["org_score"]) + np.array(json_file["baseline_score"]))] + json_file["smdl_score"][:-1]
    )
    
    # value_list = np.exp((value_list_1 - value_list_2)/1)
    value_list = value_list_1 - value_list_2
    
    values = []
    value = 0
    i = 0
    for smdl_single_mask, smdl_value in zip(S_set, value_list):
        value = value - abs(smdl_value)
        single_mask[smdl_single_mask==1] = value
        values.append(value)
        i+=1
    attribution_map = single_mask - single_mask.min()
    attribution_map = attribution_map / attribution_map.max()
    
    return attribution_map, np.array(values)

def gen_cam(image_path, mask):
    """
    Generate heatmap
        :param image: [H,W,C]
        :param mask: [H,W],range 0-1
        :return: tuple(cam,heatmap)
    """
    # Read image
    w = mask.shape[1]
    h = mask.shape[0]
    image = cv2.resize(cv2.imread(image_path), (w,h))
    # mask->heatmap
    mask = cv2.resize(mask, (int(w/20),int(h/20)))
    mask = cv2.resize(mask, (w,h))
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_VIRIDIS)  # cv2.COLORMAP_COOL
    heatmap = np.float32(heatmap)

    # merge heatmap to original image
    cam = 0.5*heatmap + 0.5*np.float32(image)
    return cam.astype(np.uint8), (heatmap).astype(np.uint8)

def norm_image(image):
    """
    Normalization image
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def simple_text_heatmap(words, scores, cmap='bwr', save_path='heatmap.png'):
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    # words = text.split()
    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    colormap = cm.get_cmap(cmap)
    
    width = 3
    words_length = 0
    for word in words:
        words_length += (len(word) + 1)

    fig, ax = plt.subplots(figsize=(width, 1.5))
    ax.axis('off')

    x = 0.0
    y = 0.0
    for word, score in zip(words, scores):
        color = colormap(norm(score))
        ax.text(x, y, word, fontsize=14, ha='left', va='center',
                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))
        x += (len(word) * width/ words_length + 0.3)
        if x > 2.5:
            x = 0
            y -= 0.3
            
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    
def get_word_saliency(json_file):
    insertion_word_scores  = [json_file["baseline_score"]] + json_file["insertion_word_score"]
    insertion_word_scores = np.array(insertion_word_scores).T
    
    regions = np.array([0] + json_file["region_area"])
    
    word_heatmap_scores = []
    for insertion_word_score in insertion_word_scores:
        auc = metrics.auc(regions, insertion_word_score)
        word_heatmap_score = auc - insertion_word_score.min()
        word_heatmap_scores.append(word_heatmap_score)
    
    return word_heatmap_scores

def visualize_explanation(vis_saliency_map, words, scores, cmap='bwr', fixed_width=6, bottom_vis=True):
    # ---- Data & aspect ratio ----
    vis_saliency_map = vis_saliency_map[:, :, ::-1]  # BGR->RGB
    h, w, _ = vis_saliency_map.shape
    img_aspect = h / w

    # Top panel height (inches) and fixed bottom panel height
    top_h_in = fixed_width * img_aspect
    bottom_h_in = 1.2
    fig_h_in = top_h_in + bottom_h_in

    # ---- Figure & gridspec (2 columns: main image + colorbar) ----
    fig = plt.figure(figsize=(fixed_width, fig_h_in))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[top_h_in, bottom_h_in],
        width_ratios=[1.0, 0.035]   # narrow colorbar on the right
    )

    # ---- Top panel: main image (keep aspect ratio, fill width) ----
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    ax0.set_aspect('equal', adjustable='box')  # Keep ratio, no stretching
    im = ax0.imshow(vis_saliency_map)

    # Separate colorbar axis: does not compress main image and avoids large side margins
    cax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cax)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.ax.tick_params(length=0, labelbottom=False, labelleft=False)
    cbar.ax.set_yticklabels([])
    
    if bottom_vis == False:
        return

    # ---- Bottom panel (span both columns, auto line wrapping) ----
    ax1 = fig.add_subplot(gs[1, :])
    ax1.set_axis_off()
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    colormap = cm.get_cmap(cmap)

    # Key: measure text width in pixels -> convert to axes coordinates
    fig.canvas.draw()                      # Ensure renderer is available
    renderer = fig.canvas.get_renderer()
    inv = ax1.transAxes.inverted()         # Pixel -> axes coordinate transform

    left_margin   = 0.02
    right_margin  = 0.98
    word_gap_axes = 0.02                   # Fixed gap between words (axes coordinates)
    line_height   = 0.28                   # Line spacing (axes coordinates)

    x, y = left_margin, 0.72               # Starting position (axes coordinates)
    for wd, sc in zip(words, scores):
        color = colormap(norm(sc))

        # Temporarily place the text object to measure its width
        t = ax1.text(x, y, wd, transform=ax1.transAxes,
                     fontsize=12, ha='left', va='center',
                     bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.25'))

        fig.canvas.draw()                  # Update to get bounding box
        bb = t.get_window_extent(renderer=renderer)          # Bounding box in pixels
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        w_axes = x1 - x0                   # Text width in axes coordinates

        # If it doesn't fit in current line: move to next line
        if x + w_axes > right_margin:
            t.set_position((left_margin, y - line_height))
            x = left_margin + w_axes + word_gap_axes
            y -= line_height
        else:
            x = x + w_axes + word_gap_axes
            
    # ====== Text colorbar ======
    divider = make_axes_locatable(ax1)
    cax_text = divider.append_axes("right", size="3%", pad=0.02)

    sm = ScalarMappable(cmap=cm.get_cmap('bwr'), norm=norm)  # Same norm as text colors
    sm.set_array([])  # Required for colorbar
    cbar_text = fig.colorbar(sm, cax=cax_text)
    cbar_text.set_ticks([])  # Remove ticks
    for spine in cbar_text.ax.spines.values():
        spine.set_visible(False)
    cbar_text.ax.tick_params(length=0)

    # Adjust margins (avoid mixing with constrained_layout)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.06, wspace=0.04, hspace=0.04)

    # plt.show()
    # Save without borders:
    # plt.savefig("panel.pdf", bbox_inches='tight', pad_inches=0)
    # plt.close()
    
def visualization_mllm(image_path, S_set, saved_json_file, save_path=None):
    if 'smdl_score' in saved_json_file:
        attribution_map, _ = add_value(S_set, saved_json_file)
        attribution_map = norm_image(attribution_map[:,:,0])
    else:
        attribution_map = S_set
        attribution_map = attribution_map - attribution_map.min()
        attribution_map = attribution_map / (attribution_map.max() + 0.00000001)
        image = cv2.imread(image_path)
        attribution_map = cv2.resize(attribution_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        attribution_map = norm_image(attribution_map)
    
    vis_saliency_map, heatmap = gen_cam(image_path, attribution_map)
     
    text = saved_json_file.get("words", None)
   
    if text:
        word_heatmap_scores = get_word_saliency(saved_json_file)
        visualize_explanation(vis_saliency_map, text, word_heatmap_scores)
    else:
        visualize_explanation(vis_saliency_map, None, None, bottom_vis=False)
    
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    plt.close()

def annotate_with_grounding_dino(image, boxes, phrases, color=(34,139,34)):
    """
    使用 Grounding DINO 格式化可视化目标检测结果

    参数:
        image (np.ndarray): 输入图像 (BGR) 格式
        boxes (np.ndarray): 检测框坐标，格式为 xyxy，形状为 (N, 4)
        phrases (List[str]): 每个检测框对应的类别标签列表

    返回:
        np.ndarray: 可视化的图像
    """
    import torch
    from torchvision.ops import box_convert
    import supervision as sv
    
    # 将坐标转换为 Torch 张量，并确保数据类型一致
    boxes = torch.tensor(boxes, dtype=torch.float32)
    
    class_ids = np.zeros(len(boxes), dtype=int)
    
    # 获取图像的宽和高
    h, w, _ = image.shape

    # 确保坐标与图像尺寸匹配
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)  # 限制x坐标在图像范围内
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)  # 限制y坐标在图像范围内

    # 将框转换为 cxcywh 格式，再转换回 xyxy 格式
    xyxy_boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy").numpy()
    
    # 使用 supervision 库来进行可视化
    detections = sv.Detections(xyxy=xyxy_boxes, class_id=class_ids)

    # 初始化监督库中的注释器
    bbox_annotator = sv.BoxAnnotator(
        color=sv.Color(r=color[0], g=color[1], b=color[2]),
        thickness=4,
        # corner_radius=2
        )
    # label_annotator = sv.LabelAnnotator(color=sv.Color(r=color[0], g=color[1], b=color[2]))
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.WHITE,             # 框颜色
        text_color=sv.Color.BLACK,        # 文字颜色
        # text_background=sv.Color.WHITE    # 文字背景
        text_scale=1.2
    )

    # 转换图像格式为 BGR（OpenCV 格式）
    # annotated_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_frame = image
    
    # 绘制边框
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # 绘制标签
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)

    return annotated_frame



def visualization_mllm_with_object(image_path, S_set, saved_json_file, save_path=None):
    if 'smdl_score' in saved_json_file:
        attribution_map, _ = add_value(S_set, saved_json_file)
    
        sensitive = max(saved_json_file["insertion_score"]) - saved_json_file["deletion_score"][-1]
        print(sensitive)
        scale = sensitive * 1.5
        
        attribution_map = norm_image(attribution_map[:,:,0])
        
        if scale < 1:
            attribution_map = attribution_map * scale
    else:
        attribution_map = S_set
        attribution_map = attribution_map - attribution_map.min()
        attribution_map = attribution_map / (attribution_map.max() + 0.00000001)
        image = cv2.imread(image_path)
        attribution_map = cv2.resize(attribution_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        attribution_map = norm_image(attribution_map)
    
    vis_saliency_map, heatmap = gen_cam(image_path, attribution_map)
    
    vis_saliency_map_w_box = annotate_with_grounding_dino(vis_saliency_map, np.array([saved_json_file["location"]]), [saved_json_file["select_category"]], color=(255,255,255))
    
    
    # word_heatmap_scores = get_word_saliency(saved_json_file)
    
    # text = saved_json_file["output_words"]
    text = saved_json_file.get("output_words", None)
    if text:
        word_mask = np.zeros(len(saved_json_file["output_words"]))

        for selected_interpretation_token_ in saved_json_file["selected_interpretation_token_id"]:
            word_mask[selected_interpretation_token_] = 1
        
        visualize_explanation_with_special_word(vis_saliency_map, text, word_mask)
    else:
         visualize_explanation_with_special_word(vis_saliency_map, None, None, bottom_vis=False)   
    
    if save_path != None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    plt.close()
 

def visualize_explanation_with_special_word(vis_saliency_map, words, word_mask, fixed_width=6, bottom_vis=True):
    # ---- Data & aspect ratio ----
    vis_saliency_map = vis_saliency_map[:, :, ::-1]  # BGR->RGB
    h, w, _ = vis_saliency_map.shape
    img_aspect = h / w

    # Top panel height (inches) and fixed bottom panel height
    top_h_in = fixed_width * img_aspect
    bottom_h_in = 1.2
    fig_h_in = top_h_in + bottom_h_in

    # ---- Figure & gridspec (2 columns: main image + colorbar) ----
    fig = plt.figure(figsize=(fixed_width, fig_h_in))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[top_h_in, bottom_h_in],
        width_ratios=[1.0, 0.035]   # narrow colorbar on the right
    )

    # ---- Top panel: main image (keep aspect ratio, fill width) ----
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    ax0.set_aspect('equal', adjustable='box')  # Keep ratio, no stretching
    im = ax0.imshow(vis_saliency_map)

    # Separate colorbar axis: does not compress main image and avoids large side margins
    cax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(im, cax=cax)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)
    cbar.ax.tick_params(length=0, labelbottom=False, labelleft=False)
    cbar.ax.set_yticklabels([])
    
    if bottom_vis == False:
        return

    # ---- Bottom panel (span both columns, auto line wrapping) ----
    ax1 = fig.add_subplot(gs[1, :])
    ax1.set_axis_off()
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    # norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    # colormap = cm.get_cmap(cmap)

    # Key: measure text width in pixels -> convert to axes coordinates
    fig.canvas.draw()                      # Ensure renderer is available
    renderer = fig.canvas.get_renderer()
    inv = ax1.transAxes.inverted()         # Pixel -> axes coordinate transform

    left_margin   = 0.02
    right_margin  = 0.98
    word_gap_axes = 0.02                   # Fixed gap between words (axes coordinates)
    line_height   = 0.28                   # Line spacing (axes coordinates)

    x, y = left_margin, 1.2               # Starting position (axes coordinates)
    for wd, sc in zip(words, word_mask):
        wd = wd.strip()
        if sc == 1:
            color = (0.8000, 1.0000, 0.8000)
            # color = colormap(norm(sc))
        else:
            color = (1, 1, 1)
            # color = (0.827, 0.827, 0.827)
            # color = (0.9, 0.9, 0.9)
        
        # Temporarily place the text object to measure its width
        t = ax1.text(x, y, wd, transform=ax1.transAxes,
                     fontsize=12, ha='left', va='center',
                     bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.25'))

        fig.canvas.draw()                  # Update to get bounding box
        bb = t.get_window_extent(renderer=renderer)          # Bounding box in pixels
        (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        w_axes = x1 - x0                   # Text width in axes coordinates

        # If it doesn't fit in current line: move to next line
        if x + w_axes > right_margin:
            t.set_position((left_margin, y - line_height))
            x = left_margin + w_axes + word_gap_axes
            y -= line_height
        else:
            x = x + w_axes + word_gap_axes
            
    # ====== Text colorbar ======
    # divider = make_axes_locatable(ax1)
    # cax_text = divider.append_axes("right", size="3%", pad=0.02)

    # sm = ScalarMappable(cmap=cm.get_cmap('bwr'), norm=norm)  # Same norm as text colors
    # sm.set_array([])  # Required for colorbar
    # cbar_text = fig.colorbar(sm, cax=cax_text)
    # cbar_text.set_ticks([])  # Remove ticks
    # for spine in cbar_text.ax.spines.values():
    #     spine.set_visible(False)
    # cbar_text.ax.tick_params(length=0)

    # # Adjust margins (avoid mixing with constrained_layout)
    # fig.subplots_adjust(left=0.02, right=0.985, top=0.985, bottom=0.06, wspace=0.04, hspace=0.04)

    # plt.show()
    # Save without borders:
    # plt.savefig("panel.pdf", bbox_inches='tight', pad_inches=0)
    # plt.close()
    