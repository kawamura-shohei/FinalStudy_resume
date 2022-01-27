import geotiff_io
import cv2
import numpy as np

def main():
    ## ファイルパス
    # 建物マスク
    infileBldMask_path = "../../sample_data01/BldMask/AtamiBldMask01.tif"
    # 被害建物の正解画像
    infileTrueAnswer_path = "../../sample_data01/Atami_AnswerTrue01.tif"
    # 無被建物の正解画像
    infileFalseAnswer_path = "../../sample_data01/Atami_AnswerFalse01.tif"
    # 手法による出力結果
    infileMyAnswer_path = "../../sample_data01/level2_result_integration/Final_result01.tif"

    ## 画像読み込み
    img01 = geotiff_io.read_geotiff(infileBldMask_path)
    imgAllBld = geotiff_io.cvtTiff2CV2_1band_to_1band(img01)
    img02 = geotiff_io.read_geotiff(infileTrueAnswer_path)
    imgTrue = geotiff_io.cvtTiff2CV2_1band_to_1band(img02)
    img03 = geotiff_io.read_geotiff(infileFalseAnswer_path)
    imgFalse = geotiff_io.cvtTiff2CV2_1band_to_1band(img03)
    img04 = geotiff_io.read_geotiff(infileMyAnswer_path)
    imgMyAnswer = geotiff_io.cvtTiff2CV2_1band_to_1band(img04)

    ## ラベリング(4連結でラベリングしなくては数が合わない)
    nLabels_All, labelImgages_All, data_All, center_All = cv2.connectedComponentsWithStatsWithAlgorithm(imgAllBld.astype("uint8"), 4, cv2.CV_16U, cv2.CCL_WU)
    nLabels_True, labelImgages_True, data_True, center_True = cv2.connectedComponentsWithStatsWithAlgorithm(imgTrue.astype("uint8"), 4, cv2.CV_16U, cv2.CCL_WU)
    nLabels_False, labelImgages_False, data_False, center_False = cv2.connectedComponentsWithStatsWithAlgorithm(imgFalse.astype("uint8"), 4, cv2.CV_16U, cv2.CCL_WU)
    nLabels_MyAnswer, labelImgages_MyAnswer, data_MyAnswer, center_MyAnswer = cv2.connectedComponentsWithStatsWithAlgorithm(imgMyAnswer.astype("uint8"), 4, cv2.CV_16U, cv2.CCL_WU)

    ## TP求める
    imgTP = cv2.bitwise_and(imgTrue, imgMyAnswer)
    geotiff_io.write_geotiff_1("../../sample_data01/Atami_TP01.tif", img01, imgTP)
    nLabels_TP, labelImgages_TP, data_TP, center_TP = cv2.connectedComponentsWithStatsWithAlgorithm(imgTP.astype("uint8"), 4, cv2.CV_16U, cv2.CCL_WU)

    ## 各種数値を求める
    all_buildings = nLabels_All - 1
    higai_buildings = nLabels_True - 1
    muhigai_buildings = nLabels_False - 1
    myanswer_buildings = nLabels_MyAnswer - 1
    TP_buildings = nLabels_TP - 1
    FN_buildings = higai_buildings - TP_buildings
    FP_buildings = myanswer_buildings - TP_buildings
    TN_buildings = all_buildings - (TP_buildings + FN_buildings + TP_buildings)
    tekigo = TP_buildings / (TP_buildings + FP_buildings)
    saigen = TP_buildings / (TP_buildings + FN_buildings)
    F_num = 2 * (tekigo * saigen) / (tekigo + saigen)
    print("---建物数---")
    print(all_buildings)
    print(higai_buildings)
    print(muhigai_buildings)
    print(myanswer_buildings)
    print("------------")
    print("---TP, FN, FP, TN---")
    print(TP_buildings)
    print(FN_buildings)
    print(FP_buildings)
    print(TN_buildings)
    print("--------------------")
    print("---適合率, 再現率, F値---")
    print(tekigo)
    print(saigen)
    print(F_num)
    print("------------------------")


if __name__ == '__main__':
        main()
