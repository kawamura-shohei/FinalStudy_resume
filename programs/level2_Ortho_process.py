from numpy.core.fromnumeric import shape
import geotiff_io
import cv2
import numpy as np

### 提案手法の粒度レベル2の処理 [画像による被害家屋候補検知] を行う ###

## 引数は下記の通り ##
# BeforeOrtho   : 災害前オルソ画像(level0ではimgOrtho_Before)
# AfterOrtho    :　災害後オルソ画像(level0ではimgOrtho_After)
# BldMask_forOrtho  : 住宅マスク(level0ではimgBldMaskだったもののlevel1でのコピー)
# DosyaMask : 土砂被害部分マスク(level1ではimg_DosyaMask)
def Ortho_detection(BeforeOrtho, AfterOrtho, BldMask_forOrtho, DosyaMask):
    ## ファイルパス ##
    # 色調補正された災害前オルソ画像
    infileBeforeHosei_path = "../../sample_data01/level2_ortho_result/AtamiOrtho_BeforeHosei.tif"
    # グレースケール化された災害前オルソ画像
    infileBeforeGray_path = "../../sample_data01/level2_ortho_result/AtamiOrtho_BeforeGray.tif"
    # 鮮鋭化された災害前オルソ画像
    infileBeforeSharp_path = "../../sample_data01/level2_ortho_result/AtamiOrtho_BeforeSharp.tif"
    # ノイズ除去された災害前オルソ画像
    infileBeforeFiltered_path = "../../sample_data01/level2_ortho_result/AtamiOrtho_BeforeFiltered.tif"
    # 災害前のエッジ画像
    infileBeforeEdge_path = "../../sample_data01/level2_ortho_result/AtamiOrtho_BeforeEdge.tif"
    # グレースケール化された災害後オルソ画像
    infileAfterGray_path = "../../sample_data01/level2_ortho_result/AtamiOrtho_AfterGray.tif"
    # ノイズ除去された災害後オルソ画像
    infileAfterFiltered_path = "../../sample_data01/level2_ortho_result/AtamiOrtho_AfterFiltered.tif"
    # 災害後のエッジ画像
    infileAfterEdge_path = "../../sample_data01/level2_ortho_result/AtamiOrtho_AfterEdge.tif"
    # 正規化されたcos類似度を格納した結果画像
    infile_preResult_path = "../../sample_data01/level2_ortho_result/preResult.tif"
    # 画像による被害候補家屋
    Ortho_result_path = "../../sample_data01/level2_ortho_result/Ortho_result01.tif"

    # オルソ画像のための位置情報
    img13 = geotiff_io.read_geotiff("../../sample_data01/level1_result/AtamiOrtho_After01.tif")

    # ラプラシアンフィルタのカーネル
    kernel = np.array([
                    [1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1]
                    ], np.float32)

    # ## 災害前のオルソ画像への前処理 ##
    # # 色調補正
    #     # HLS表色系に変換
    # imgBeforeOrtho_HLS = cv2.cvtColor(BeforeOrtho, cv2.COLOR_BGR2HLS)
    # imgAfterOrtho_HLS = cv2.cvtColor(AfterOrtho, cv2.COLOR_BGR2HLS)
    #     # 輝度から平均と標準偏差を求める
    # imgBeforeOrtho_HLS_mean, imgBeforeOrtho_HLS_std = cv2.meanStdDev(imgBeforeOrtho_HLS[:,:,1])
    # imgAfterOrtho_HLS_mean, imgAfterOrtho_HLS_std = cv2.meanStdDev(imgAfterOrtho_HLS[:,:,1])
    #     # 補正
    # BeforeOrtho_correction = (imgAfterOrtho_HLS_std / imgBeforeOrtho_HLS_std) * (BeforeOrtho - imgBeforeOrtho_HLS_mean) + imgAfterOrtho_HLS_mean
    # BeforeOrtho_correction = np.where(255 < BeforeOrtho_correction, 255, BeforeOrtho_correction)
    # BeforeOrtho_correction = np.where(BeforeOrtho_correction < 0, 0, BeforeOrtho_correction)
    # geotiff_io.write_geotiff_3(infileBeforeHosei_path, img13, BeforeOrtho_correction)

    # # グレースケール化
    # imgBeforeOrtho_Gray = cv2.cvtColor(BeforeOrtho_correction.astype("uint8"), cv2.COLOR_BGR2GRAY)
    # geotiff_io.write_geotiff_1(infileBeforeGray_path, img13, imgBeforeOrtho_Gray)

    # # 鮮鋭化
    # sharp_kernel = make_sharp_kernel(9)
    # imgBeforeOrtho_Sharp = cv2.filter2D(imgBeforeOrtho_Gray, -1, sharp_kernel, delta=-200)
    # geotiff_io.write_geotiff_1(infileBeforeSharp_path, img13, imgBeforeOrtho_Sharp)

    # # ノイズ除去
    #     # バイラテラルフィルタを2回適用(パラメータは引数の通り)
    # imgBeforeOrtho_bilateral01 = cv2.bilateralFilter(imgBeforeOrtho_Sharp, 15, 20, 20)
    # imgBeforeOrtho_bilateral02 = cv2.bilateralFilter(imgBeforeOrtho_bilateral01, 15, 20, 20)
    # geotiff_io.write_geotiff_1(infileBeforeFiltered_path, img13, imgBeforeOrtho_bilateral02)


    # ## 前処理された災害前オルソ画像のエッジ抽出
    # imgBeforeOrtho_Filter = cv2.filter2D(imgBeforeOrtho_bilateral02, -1, kernel)
    # geotiff_io.write_geotiff_1(infileBeforeEdge_path, img13, imgBeforeOrtho_Filter)
    img14 = geotiff_io.read_geotiff(infileBeforeEdge_path)
    imgBeforeOrtho_Edge = geotiff_io.cvtTiff2CV2_1band_to_1band(img14)


    # ## 災害後のオルソ画像への前処理 ##
    # # グレースケール化
    # imgAfterOrtho_Gray = cv2.cvtColor(AfterOrtho, cv2.COLOR_BGR2GRAY)
    # geotiff_io.write_geotiff_1(infileAfterGray_path, img13, imgAfterOrtho_Gray)

    # # ノイズ除去
    #     # バイラテラルフィルタを2回適用(パラメータは引数の通り)
    # imgAfterOrtho_bilateral01 = cv2.bilateralFilter(imgAfterOrtho_Gray, 15, 20, 20)
    # imgAfterOrtho_bilateral02 = cv2.bilateralFilter(imgAfterOrtho_bilateral01, 15, 20, 20)
    # geotiff_io.write_geotiff_1(infileAfterFiltered_path, img13, imgAfterOrtho_bilateral02)


    # ## 前処理された災害後オルソ画像のエッジ抽出
    # imgAfterOrtho_Filter = cv2.filter2D(imgAfterOrtho_bilateral02, -1, kernel)
    # geotiff_io.write_geotiff_1(infileAfterEdge_path, img13, imgAfterOrtho_Filter)
    img15 = geotiff_io.read_geotiff(infileAfterEdge_path)
    imgAfterOrtho_Edge = geotiff_io.cvtTiff2CV2_1band_to_1band(img15)


    ## 災害前後で類似度の低い建物領域を検知 ##
    # ラベリング
    nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(BldMask_forOrtho.astype("uint8"))

    # 最終結果になる画像を黒画像として定義
    img_level2_Ortho_result00 = np.zeros((BldMask_forOrtho.shape[0], BldMask_forOrtho.shape[1]))

    # 土砂被害部分と重なる建物領域を検知
    imgDosyaBld_Mask_proto01 = cv2.bitwise_and(BldMask_forOrtho, DosyaMask).astype("uint8")

    # 被害可能性の高い建物のラベル番号を格納するリスト
    i_num = []

    # cos類似度を格納するリスト
    cos_ret = []

    # 災害前後のエッジで類似度を計算
    for i in range(1, nLabels):
        print(i)
        i_Building = np.where(labelImages == i, 255, 0).astype("uint8") # i番目の建物領域
        i_Mask = cv2.bitwise_and(imgDosyaBld_Mask_proto01, i_Building)  # i番目の建物領域で土砂被害部分と重なっている領域

        if 255 in i_Mask:
            print("------------")
            i_num.append(i) # 被害可能性の高い建物のラベル番号を追加
            i_Building_BohChoh = cv2.dilate(i_Building, kernel, iterations=5)
            # cos類似度
            ret = cv2.matchTemplate(imgBeforeOrtho_Edge, imgAfterOrtho_Edge, cv2.TM_CCORR_NORMED, mask=i_Building_BohChoh)
            print(ret)
            cos_ret.append(ret[0][0]) # cos類似度を追加
            print("------------")

    # 類似度を正規化
    max_val = np.nanmax(cos_ret)
    min_val = np.nanmin(cos_ret)
    print("----------")
    print(max_val)
    print(min_val)
    print("----------")
    ret_normal = geotiff_io.normalized(cos_ret, max_val, min_val)

    # 正規化された類似度を各建物領域に入れる
    k = 0
    for j in i_num:
        img_level2_Ortho_result00 = np.where(labelImages == j, ret_normal[k], img_level2_Ortho_result00)
        k += 1
    
    # # 閾値処理
    img_level2_Ortho_result00 = np.where(img_level2_Ortho_result00 == 255, 0, img_level2_Ortho_result00)
    geotiff_io.write_geotiff_1(infile_preResult_path, img13, img_level2_Ortho_result00)
    # img16 = geotiff_io.read_geotiff(infile_preResult_path)
    # img_level2_Ortho_result00 = geotiff_io.cvtTiff2CV2_1band_to_1band(img16)
    # 類似度の閾値は [180上流部が] [下流部が160]
    th01, img_level2_Ortho_result01 = cv2.threshold(img_level2_Ortho_result00, 180, 255, cv2.THRESH_TOZERO_INV)
    th02, img_level2_Ortho_result = cv2.threshold(img_level2_Ortho_result01, 1, 255, cv2.THRESH_BINARY)
    geotiff_io.write_geotiff_1(Ortho_result_path, img13, img_level2_Ortho_result)

    # return img_level2_Ortho_result


# 鮮鋭化フィルタ生成関数
def make_sharp_kernel(k: int):
  return np.array([
    [-k / 9, -k / 9, -k / 9],
    [-k / 9, 1 + 8 * k / 9, k / 9],
    [-k / 9, -k / 9, -k / 9]
  ], np.float32)