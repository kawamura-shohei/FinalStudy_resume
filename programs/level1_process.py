import geotiff_io
import level2_DSM_process
import level2_DosyaMask_process
import level2_Ortho_process
import level2_ResultIntegration_process

### 提案手法の粒度レベル1の処理 [土砂災害による流出家屋検知] を行う ###

## 引数は下記の通り ##
# BeforeDSM : 災害前DSM(level0ではimgDSM_Before)
# AfterDSM  : 災害後DSM(level0ではimgDSM_After)
# BeforeOrtho   : 災害前オルソ画像(level0ではimgOrtho_Before)
# AfterOrtho    : 災害後オルソ画像(level0ではimgOrtho_After)
# BldMask   : 住宅マスク(level0ではimgBldMask)
def house_spill_detection(BeforeDSM, AfterDSM, BeforeOrtho, AfterOrtho, BldMask):
    # 住宅マスクのコピー
    BldMask_forDSM = BldMask.copy()
    BldMask_forOrtho = BldMask.copy()

    ## 提案手法の粒度レベル2の処理 [DSM差分による被害家屋候補検知] を行う ##
    # img_DSM_detection = level2_DSM_process.DSM_detection(BeforeDSM, AfterDSM, BldMask_forDSM)
    img11 = geotiff_io.read_geotiff("../../sample_data01/level2_DSM_result/DSM_result01.tif")
    img_DSM_detection = geotiff_io.cvtTiff2CV2_1band_to_1band(img11)
    print(img_DSM_detection.shape)

    ## 提案手法の粒度レベル2の処理 [土砂被害マスク作成] を行う ##
    # img_DosyaMask = level2_DosyaMask_process.Make_DosyaMask(AfterOrtho)
    img12 = geotiff_io.read_geotiff("../../sample_data01/level2_Mask_result/DosyaMask01.tif")
    img_DosyaMask = geotiff_io.cvtTiff2CV2_1band_to_1band(img12)
    print(img_DosyaMask.shape)

    ## 提案手法の粒度レベル2の処理 [画像による被害家屋候補検知] を行う ##
    # img_Ortho_detection = level2_Ortho_process.Ortho_detection(BeforeOrtho, AfterOrtho, BldMask_forOrtho, img_DosyaMask)
    img17 = geotiff_io.read_geotiff("../../sample_data01/level2_ortho_result/Ortho_result01.tif")
    img_Ortho_detection = geotiff_io.cvtTiff2CV2_1band_to_1band(img17)
    print(img_Ortho_detection.shape)

    ## 提案手法の粒度レベル2の処理 [結果の統合] を行う ##
    img_level1_result = level2_ResultIntegration_process.Result_Integration(img_DSM_detection, img_DosyaMask, img_Ortho_detection)

    # return img_level1_result