
ώ
6
Add_Prev_Layer(0J  5   FM   F` p 
4
LayerNorm_1(0J  5   FM   F`  
<
Q(0:/   5   FE   LM   F`   ₯   L
<
K(0:/   5   FE   LM   F`   ₯   L
<
V(0:/   5   FE   LM   F`   ₯   L
8
K_cache(0J%   5   FM  L`   ₯   L
8
V_cache(0J%   5   FM  L`   ₯   L
?

MHA_GEMM_1(0B)   5   FE  LM I`   
2
SOFTMAX	(0J  5 IM I`  
4
	DropOut_1
(0J  5 IM I`  
?

MHA_GEMM_2(0B)   5  LE IM   F`   
K
	PROJ_GEMM(0:6   5   FE   LM   F`hu   F   ₯   L
2
	DropOut_2	(0J  5   FM   F`  
4
Add_1
(0R#  5   FM   F`  υ   F
4
LayerNorm_2(0J  5   FM   F`  
?
FFN0(0:/p  5   FE  ΰLM  ΰF`p  ₯  ΰL
-
GeLU(0Jp 5  ΰFM  ΰF`p 
F
FFN1(0:6 p 5  ΰFE  ΰLM   F`hu   F p ₯  ΰL
2
	DropOut_3(0J  5   FM   F`  
4
Add_2(0R#  5   FM   F`  υ   F7-   0=   FE  ΰLPZAdd_Prev_LayerbLayerNorm_1 %0=   FE   FPZLayerNorm_1bQ %0=   FE   FPZLayerNorm_1bK %0=   FE   FPZLayerNorm_1bV !0=   FE   FPZKbK_cache !0=   FE   FPZVbV_cache $0=   FE   FPZQb
MHA_GEMM_1 *0=  LE  LPZK_cacheb
MHA_GEMM_1 *	0= IE IP	Z
MHA_GEMM_1bSOFTMAX )	
0= IE IP
ZSOFTMAXb	DropOut_1 ,
0= IE IPZ	DropOut_1b
MHA_GEMM_2 *0=  LE  LPZV_cacheb
MHA_GEMM_2 ,0=   FE   FPZ
MHA_GEMM_2b	PROJ_GEMM 0-   0=   FE   FPZ	PROJ_GEMMb	DropOut_2 ,-   0=   FE   FPZ	DropOut_2bAdd_1 .-   0=   FE   FPZAdd_1bLayerNorm_2 1-   0=   FE  ΰLPZAdd_Prev_LayerbAdd_1 (0=   FE   FPZLayerNorm_2bFFN0 &-   0=  ΰFE  ΰFPZFFN0bGeLU &-   0=  ΰFE  ΰFPZGeLUbFFN1 +-   0=   FE   FPZFFN1b	DropOut_3 ,-   0=   FE   FPZ	DropOut_3bAdd_2 (-   0=   FE   FPZAdd_1bAdd_2 . χ6 -ΝΜ$QUΝΜ?" ­ffΦK²DP’qK(   @  ΐA  ?% θ G-Γd*<EτύT=M(a&>U ;D":   ( X`xpx*o:΄