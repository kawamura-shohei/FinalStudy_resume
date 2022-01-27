import geotiff_io
import level1_process

### 提案手法の粒度レベル0の処理を行う ###

def main():
    ## ファイルパス ##
    # 災害前後DSM
    inflileDSM_Before_path = "../../sample_data01/level1_result/AtamiDSM_Before01.tif"
    inflileDSM_After_path = "../../sample_data01/level1_result/AtamiDSM_After01.tif"
    # 災害前後オルソ画像
    inflileOrtho_Before_path = "../../sample_data01/level1_result/AtamiOrtho_Before01.tif"
    inflileOrtho_After_path = "../../sample_data01/level1_result/AtamiOrtho_After01.tif"
    # 住宅マスク
    infileBldMask_path = "../../sample_data01/BldMask/AtamiBldMask01.tif"

    ## 画像読み込み ##
    # 災害前後DSM
    img01 = geotiff_io.read_geotiff(inflileDSM_Before_path)
    img02 = geotiff_io.read_geotiff(inflileDSM_After_path)
    imgDSM_Before = geotiff_io.cvtTiff2CV2_1band_to_1band(img01)
    imgDSM_After = geotiff_io.cvtTiff2CV2_1band_to_1band(img02)
    # 災害前後オルソ画像
    img03 = geotiff_io.read_geotiff(inflileOrtho_Before_path)
    img04 = geotiff_io.read_geotiff(inflileOrtho_After_path)
    imgOrtho_Before = geotiff_io.cvtTiff2CV2_3band(img03)
    imgOrtho_After = geotiff_io.cvtTiff2CV2_3band(img04)
    # 住宅マスク
    img05 = geotiff_io.read_geotiff(infileBldMask_path)
    imgBldMask = geotiff_io.cvtTiff2CV2_1band_to_1band(img05)

    ## 提案手法の粒度レベル1の処理 [土砂災害による流出家屋検知] を行う ##
    # img_level0_result = 
    level1_process.house_spill_detection(imgDSM_Before, imgDSM_After, imgOrtho_Before, imgOrtho_After, imgBldMask)


if __name__ == '__main__':
        main()
