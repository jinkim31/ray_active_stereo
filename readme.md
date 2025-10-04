decode map : shape ={ N,H,W,2] (x,y 순서)  value =[0,1] 
decode mask : shape = [N,H,W]



# 실행 절차

## 1. Phase Shift Decode

1. `scripts/intrinsic_decode_images.py` 들어가기
2. 아래 부분 수정
   - `calib_images_dir`: 카메라 하나의 Phase shift 패턴 캡쳐 들어있는 폴더.
   - `result_dir`: 디코드 맵 저장될 폴더.
   ```python
   # Arguments
   calib_images_dir = '../datasets/intrinsic/192.168.1.210'
   result_dir = '../results/intrinsic_decode_maps/192.168.1.210'
   lcd_resolution = [3840, 2160]
   ```
3. `scripts/intrinsic_decode_images.py` 실행 
4. Arguments `192.168.1.200` 으로도 2~3 실행

## Intrinsic Calibration

1. 