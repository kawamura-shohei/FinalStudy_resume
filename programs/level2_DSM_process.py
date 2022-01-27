import geotiff_io
import cv2
import numpy as np

### 提案手法の粒度レベル2の処理 [土砂被害マスク作成] を行う ###

## 引数は下記の通り ##
# BeforeDSM : 災害前DSM(level0ではimgDSM_Before)
# AfterDSM  : 災害後DSM(level0ではimgDSM_After)
# BldMask_forDSM    : 住宅マスク(level0ではimgBldMaskだったもののlevel1でのコピー)
def DSM_detection(BeforeDSM, AfterDSM, BldMask_forDSM):
    ## ファイルパス ##
    # 建物DSM差分
    DSM_sabun_path = "../../sample_data01/level2_DSM_result/DSMsabun_Bld01.tif"
    # 正規化された建物DSM差分
    normalized_path = "../../sample_data01/level2_DSM_result/DSMsabun_normal.tif"
    # 二値化された建物DSM差分
    binary_path = "../../sample_data01/level2_DSM_result/DSMsabun_bin.tif"
    # 二値化される前のDSM差分による被害候補家屋
    pre_DSM_result_path = "../../sample_data01/level2_DSM_result/DSMsabun_pre_fin.tif"
    # DSM差分による被害候補家屋
    DSM_result_path = "../../sample_data01/level2_DSM_result/DSM_result01.tif"

    ## 建物DSM差分取得 ##
    img06 = geotiff_io.read_geotiff(DSM_sabun_path)
    img_DSM_sabun_Bld = geotiff_io.cvtTiff2CV2_1band_to_1band(img06)

    ## 標高差分[大]の建物領域検知 ##
    # 正規化
    img_DSM_sabun_Bld_normal = geotiff_io.normalized(img_DSM_sabun_Bld, img_DSM_sabun_Bld.max(), img_DSM_sabun_Bld.min())
    geotiff_io.write_geotiff_1(normalized_path, img06, img_DSM_sabun_Bld_normal)

    # 二値化    [上流部での閾値は136]  [下流部での閾値は136]
    th_first, img_DSM_binary = cv2.threshold(img_DSM_sabun_Bld_normal, 136, 255, cv2.THRESH_BINARY)
    geotiff_io.write_geotiff_1(binary_path, img06, img_DSM_binary)

    # ラベリング(建物マスクの解像度を合わせつつ)
    nLabels, labelImgages, data, center = cv2.connectedComponentsWithStats(BldMask_forDSM.astype("uint8"))
    img_result = labelImgages.copy()

    # 二値化画像における標高差分[大]の建物領域が255だと以下の処理に不都合なので128(0か255でなければよい)に変更
    img_DSM_binary_255to128 = cv2.resize(np.where(img_DSM_binary == 255, 128, img_DSM_binary), dsize=(BeforeDSM.shape[1], BeforeDSM.shape[0])).astype("uint8")

    # 建物領域中を占める128の画素値の割合を格納する
    list_128 = []

    # 建物領域ごとに標高差分[大]領域の割合を求める
    for i in range(1, nLabels):
        print(i)
        # i番目のラベルの建物領域マスクを作成
        i_Building = np.where(labelImgages == i, 255, 0).astype("uint8")

        # マスク処理
        i_Bld_And = cv2.bitwise_and(i_Building, img_DSM_binary_255to128)

        # 建物領域中の128の画素値の割合
        i_percent = np.count_nonzero(i_Bld_And == 128) * 100 / data[i][4]
        print(i_percent)
        list_128.append(i_percent)
    
    # list_128を正規化
    list_128_normal = geotiff_io.normalized(list_128, np.max(list_128), np.min(list_128))

    # 割合を書き込む
    for j in range(1, nLabels):
        img_result = np.where(labelImgages == j, list_128_normal[j-1], img_result)
    geotiff_io.write_geotiff_1(pre_DSM_result_path, img06, img_result)

    # 二値化    [上流部での閾値は100]  [下流部での閾値は100]
    th_second, img_level2_DSM_result = cv2.threshold(img_result, 100, 255, cv2.THRESH_BINARY)

    geotiff_io.write_geotiff_1(DSM_result_path, img06, img_level2_DSM_result)
    # return img_level2_DSM_result