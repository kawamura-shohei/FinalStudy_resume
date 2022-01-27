import geotiff_io
import cv2
import numpy as np

## Irfanviewで編集したものから被害・無被害家屋を抜き出す ##

def main():
    ## ファイルパス
    # 位置情報のパス
    inflileDSM_After_path = "../../sample_data02/level1_result/AtamiDSM_After02.tif"
    # 位置情報のない正解画像のパス
    infileNonXY_Answer_path = "../../sample_data02/Atami_TrueFalse02.tif"
    # 無被害家屋のパス
    outfileXY_FalseAnswer_path = "../../sample_data02/Atami_AnswerFalse02.tif"
    # 被害家屋のパス
    outfileXY_TrueAnswer_path = "../../sample_data02/Atami_AnswerTrue02.tif"

    # 画像読み込み
    img01 = geotiff_io.read_geotiff(inflileDSM_After_path)
    img02 = geotiff_io.read_geotiff(infileNonXY_Answer_path)
    imgOrigin = geotiff_io.cvtTiff2CV2_1band_to_1band(img02)

    # 各色
    gray_color = 128
    black_color = 0
    white_color = 255

    # 被害家屋を抜き出す
    imgFalse = imgOrigin.copy()
    imgFalse[np.where(imgFalse == white_color)] = black_color
    imgFalse[np.where(imgFalse == gray_color)] = white_color
    # imgFalse_Gray = cv2.cvtColor(imgFalse, cv2.COLOR_BGR2GRAY)
    th01, imgFalse_Result = cv2.threshold(imgFalse, 254, 255, cv2.THRESH_BINARY)
    print(th01)
    geotiff_io.write_geotiff_1(outfileXY_FalseAnswer_path, img01, imgFalse_Result)

    # 無被害家屋を抜き出す
    imgTrue = imgOrigin.copy()
    imgTrue[np.where(imgTrue == gray_color)] = black_color
    th02, imgTrue_Result = cv2.threshold(imgTrue, 254, 255, cv2.THRESH_BINARY)
    print(th02)
    geotiff_io.write_geotiff_1(outfileXY_TrueAnswer_path, img01, imgTrue_Result)

if __name__ == '__main__':
        main()
