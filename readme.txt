1, 2, 3 -> 右
11, 22, 33 -> 左
! -> 有墊片
rb -> 藍/紅
gp -> 洋紅/綠
墊片 0.75 mm
LLA厚度 2.1 mm
距離L 100 mm
水平距離x 30 mm

plot_viewing_zone.py 用法
是否用墊片(預設不用):
--shim without  -> read files without !
--shim with     -> read files with !

選擇要哪些顏色(也只會plot選擇的顏色以及對應總強度):
R
G
B
R+G
R+B
G+B
R+G+B

Examples:
python .\plot_viewing_zone.py
python .\plot_viewing_zone.py --shim with --integrate R+G+B
python .\plot_viewing_zone.py --group rb --shim with --integrate R+B G+B --out .\image\rb_with_shim.png
python .\plot_viewing_zone.py --group both --integrate R G B R+G+B
