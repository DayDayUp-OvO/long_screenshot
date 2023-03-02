import time
import typing as t
import cv2
import pyautogui
import numpy as np
from PIL import ImageGrab, Image


def merge_image_with_match_template(original_image: np.ndarray, target_image: np.ndarray, only_offset: bool = False) \
        -> t.Union[np.ndarray, int]:
    """
    通过模板匹配法 合并两张图像
    !!! 输入图像必须是rgb通道图像
    !!! target_image至少前10%部分必须包含在original_image中
    !!! 输出也是rbg三通道图像
    """
    # 图像灰度
    gray_a = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)[:gray_a.shape[0], :]
    # 去除边界非目标因素的影响  头部去除5%  两边各去除10%
    gray_b_resize = gray_b[int(gray_b.shape[0] * 0.05): int(gray_b.shape[0] * 0.2),
                    int(gray_b.shape[1] * 0.1):int(gray_b.shape[1] * 0.9)]
    res = cv2.matchTemplate(gray_b_resize, gray_a, cv2.TM_SQDIFF_NORMED)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)
    if min_val < 0.01:
        if only_offset:
            return min_loc[1] - int(target_image.shape[0] * 0.05)
        stack_image = np.vstack((original_image[:min_loc[1], :], target_image[int(gray_b.shape[0] * 0.05):, :]))
        return stack_image
    else:
        raise Exception('找不到匹配目标')


def long_screenshot_with_scroll(left: int, top: int, width: int, height: int, scroll_distance: int = -1) -> np.ndarray:
    """
    滚动长截图
    scrool_distance是默认第一次滚动长度 / 100
    """
    import pyautogui
    from PIL import ImageGrab
    return_distance = 0
    need_stack_image_list = [
        np.asarray(ImageGrab.grab(bbox=(left, top,
                                        left + width,
                                        top + height)))
    ]

    # 获取所有截图数据
    for _ in range(30):
        pyautogui.moveTo(left + width // 2,
                         top + height // 2)
        pyautogui.scroll(scroll_distance * 100)
        pyautogui.moveTo(1, 1)
        time.sleep(0.2)
        return_distance -= scroll_distance
        img = np.asarray(ImageGrab.grab(bbox=(left, top,
                                              left + width,
                                              top + height)))
        if np.array_equal(need_stack_image_list[-1], img):
            # 如果两次截图完全一致  则停止截图
            break
        else:
            if _ == 0:
                # 如果是第一次滚动截图 较小滚动判断以后每次的滚动距离
                # 截去边缘区域防止因非表格区域不滚动导致的无法匹配
                after_cut_img = img[int(img.shape[0] * 0.05):int(img.shape[0] * 0.8),
                                int(img.shape[1] * 0.2): int(img.shape[1] * 0.8), :]
                move_offset = merge_image_with_match_template(
                    need_stack_image_list[-1][int(img.shape[0] * 0.05):,
                    int(img.shape[1] * 0.2): int(img.shape[1] * 0.8), :], after_cut_img, True)
                # 计算出每个像素点偏移量对应的scroll距离，再乘总高度算出全部需要移动的距离，再取60%
                # (-1 / 高度偏移量) * 高度 * 0.6
                scroll_distance = round(-0.6 * img.shape[0] / move_offset)
            need_stack_image_list.append(img)
    pyautogui.moveTo(left + width // 2,
                     top + height // 2)
    pyautogui.scroll(return_distance * 100)
    pyautogui.moveTo(1, 1)

    # 拼图
    long_screenshot = need_stack_image_list[-1]
    for one_image in need_stack_image_list[-2::-1]:
        long_screenshot = merge_image_with_match_template(one_image, long_screenshot)
    return long_screenshot


if __name__ == '__main__':
    time.sleep(3)
    Image.fromarray(long_screenshot_with_scroll(871, 164, 1193-871, 918-164)).save('1.png')
