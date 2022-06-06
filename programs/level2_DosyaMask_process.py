import imp
import geotiff_io
import cv2
import numpy as np

### 提案手法の粒度レベル2の処理 [土砂被害マスク作成] を行う ###

## 引数は下記の通り ##
# AfterOrtho    :　災害後オルソ画像(level0ではimgOrtho_After)
def Make_DosyaMask(AfterOrtho):
    ## ファイルパス ##
    # 災害後傾斜図
    infileSlope_path = "../../sample_data01/level1_result/Atami_Slope01.tif"
    # MeanShift後の災害後傾斜図
    infileSlopeMS_path = "../../sample_data01/level2_Mask_result/SlopeMS_32_32.tif"
    # Lab表色系の災害後オルソ画像
    infileOrthoLab_path = "../../sample_data01/level2_Mask_result/OrthoLab.tif"
    # MeanShift後の災害後オルソ画像
    infileOrthoMS_path = "../../sample_data01/level2_Mask_result/OrthoMS_32_32.tif"
    # a値の災害後オルソ画像
    infileOrtho_a_path = "../../sample_data01/level2_Mask_result/Ortho_a.tif"
    # a値と傾斜の論理積マスク
    infileAND_path = "../../sample_data01/level2_Mask_result/Slope_a_AND.tif"
    # クロージングされたマスク
    infileClose_path = "../../sample_data01/level2_Mask_result/Slope_a_Close.tif"
    # オープニングされたマスク
    infileOpen_path = "../../sample_data01/level2_Mask_result/Slope_a_Open.tif"
    # 土砂被害部分マスク
    outfileDosyaMask_path = "../../sample_data01/level2_Mask_result/DosyaMask01.tif"


    ## 画像読み込み ##
    # 災害後傾斜図
    img07 = geotiff_io.read_geotiff(infileSlope_path)
    imgSlope = geotiff_io.cvtTiff2CV2_1band_to_3band(img07)
    # オルソ画像のための位置情報
    img08 = geotiff_io.read_geotiff("../../sample_data01/level1_result/AtamiOrtho_After01.tif")
    

    ## 傾斜のマスク作成 ##
    # MeanShiftで領域分割
    # imgSlope_MS_32_32 = cv2.pyrMeanShiftFiltering(imgSlope, 32, 32)
    # geotiff_io.write_geotiff_1(infileSlopeMS_path, img07, cv2.cvtColor(imgSlope_MS_32_32, cv2.COLOR_BGR2GRAY))
    # # 二値化はImageJで行う
    img09 = geotiff_io.read_geotiff("../../sample_data01/level2_Mask_result/Slope_Mask.tif")
    imgSlope_Mask = geotiff_io.cvtTiff2CV2_1band_to_1band(img09)
    


    ## a値のマスク作成 ##
    # Lab表色系に変換
    # imgAfterOrtho_Lab = cv2.cvtColor(AfterOrtho, cv2.COLOR_BGR2Lab)
    # geotiff_io.write_geotiff_3(infileOrthoLab_path, img08, imgAfterOrtho_Lab)
    # a値についてMeanShiftで領域分割
    # imgAfterOrtho_LabMS_32_32 = cv2.pyrMeanShiftFiltering(imgAfterOrtho_Lab, 32, 32)
    # geotiff_io.write_geotiff_3(infileOrthoMS_path, img08, imgAfterOrtho_LabMS_32_32)
    # a値を取り出す
    # imgAfterOrtho_L, imgAfterOrtho_a, imgAfterOrtho_b = cv2.split(imgAfterOrtho_LabMS_32_32)
    # geotiff_io.write_geotiff_1(infileOrtho_a_path, img08, imgAfterOrtho_a)
    # # 二値化はImageJで行う
    img10 = geotiff_io.read_geotiff("../../sample_data01/level2_Mask_result/Ortho_a_Mask.tif")
    imgOrtho_Mask = geotiff_io.cvtTiff2CV2_1band_to_1band(img10)
    

    ## マスク統合 ##
    # 論理積(解像度を合わせつつ)
    imgAND_Mask = cv2.bitwise_and(imgSlope_Mask, imgOrtho_Mask)
    # geotiff_io.write_geotiff_1(infileAND_path, img08, imgAND_Mask)
    # カーネル
    kernel = np.array([
        [1, 1, 1], 
        [1, 1, 1], 
        [1, 1, 1],
    ], dtype=np.uint8)
    # クロージング [上流部での回数は15]  [下流部での回数は12]
    th, imgAND_Mask_bin = cv2.threshold(imgAND_Mask, 128, 255, cv2.THRESH_BINARY) # 二値化しておかないと結果がおかしくなっちゃう
    img_delation_close = cv2.dilate(imgAND_Mask_bin, kernel, iterations=15)
    img_erosion_close = cv2.erode(img_delation_close, kernel, iterations=14)
    geotiff_io.write_geotiff_1(infileClose_path, img08, img_erosion_close)
    # オープニング [上流部での回数は15]  [下流部での回数は20]
    img_erosion_open = cv2.erode(img_erosion_close, kernel, iterations=15)
    img_delation_open = cv2.dilate(img_erosion_open, kernel, iterations=15)
    geotiff_io.write_geotiff_1(infileOpen_path, img08, img_delation_open)
    # ラベリング(面積が最大の領域をマスクとする, 背景ラベルが最大の面積の場合は除外, 最大面積の領域以外を黒で塗りつぶす)
    nLabels, labelImgages, data, center = cv2.connectedComponentsWithStats(img_delation_open)
    data[:,4][0] = 0
    max_index = np.argmax(data[:,4])
    # gdalとnumpyで原点が異なるためか、上下反転してしまうためcv2.flipを使用
    img_level2_DosyaMask_result = np.where(labelImgages != max_index, 0, 255)

    geotiff_io.write_geotiff_1(outfileDosyaMask_path, img08, img_level2_DosyaMask_result)
    # return img_level2_DosyaMask_result
