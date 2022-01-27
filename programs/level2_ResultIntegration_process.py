import geotiff_io
import cv2
import numpy as np
import math

def Result_Integration(img_DSM_detection, img_DosyaMask, img_Ortho_detection):
    ## ファイルパス
    # 輪郭抽出された土砂被害部分マスク
    out_path = "../../sample_data01/level2_result_integration/Rinkaku_DosyaMask01.tif"
    result_path = "../../sample_data01/level2_result_integration/DSM_Fix.tif"
    final_path = "../../sample_data01/level2_result_integration/Final_result01.tif"

    # オルソ画像のための位置情報
    img00 = geotiff_io.read_geotiff("../../sample_data01/level2_Mask_result/DosyaMask01.tif")
    img_DosyaMask_3ch = geotiff_io.cvtTiff2CV2_1band_to_3band(img00)

    # 輪郭抽出
    contours, hierarchy = cv2.findContours(img_DosyaMask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # 輪郭描画
    cv2.polylines(img_DosyaMask_3ch, contours[0], True, (0, 0, 255), 5)
    
    # DSM差分で検知した家屋にラベリング
    nLabels, labelImgages, data, center = cv2.connectedComponentsWithStats(cv2.resize(img_DSM_detection, dsize=(img_DosyaMask.shape[1], img_DosyaMask.shape[0])).astype("uint8"))
    imgDSM_Fix = labelImgages.copy()

    # DSM差分で検知した家屋を書き込み
    img_DosyaMask_3ch[np.where(labelImgages != 0)] = [0, 255, 0]

    # DSM差分で検知した家屋の重心と, 土砂マスクとの距離を計測して描画
    for i in range(1, nLabels):
        dd = 2147483647 # とりあえずの最短距離
        x3 = center[i][0]
        y3 = center[i][1]

        for j in range(len(contours[0])):
            x4 = contours[0][j][0][0]
            y4 = contours[0][j][0][1]
            d = get_distance(x3, y3, x4, y4)
            # 最短距離の更新
            if d < dd:
                dd = d
                x4min = x4
                y4min = y4
        
        # 最短距離を描画
        print(dd)
        cv2.line(img_DosyaMask_3ch, (int(x3), int(y3)), (int(x4min), int(y4min)), (255, 0, 0), thickness=5)

        # 土砂から遠い建物の領域を塗りつぶす [上流部の閾値は200] [下流部の閾値は200]
        if 200 < dd:
            imgDSM_Fix = np.where(i == labelImgages, 0, imgDSM_Fix)
        else:
            imgDSM_Fix = np.where(i == labelImgages, 255, imgDSM_Fix)

    geotiff_io.write_geotiff_3(out_path, img00, img_DosyaMask_3ch)
    geotiff_io.write_geotiff_1(result_path, img00, imgDSM_Fix)
    # img18 = geotiff_io.read_geotiff(result_path)
    # imgDSM_Fix = geotiff_io.cvtTiff2CV2_1band_to_1band(img18)
    print(imgDSM_Fix.shape)

    img_final_result = cv2.bitwise_or(imgDSM_Fix.astype("uint8"), img_Ortho_detection.astype("uint8"))
    geotiff_io.write_geotiff_1(final_path, img00, img_final_result)

    # return img_final_result


# 二点のユークリッド距離を求める関数
def get_distance(x1, y1, x2, y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d