

7
Add_Prev_Layer(0J@ 5  FM  F`@ΰ 
4
LayerNorm_1(0J@ 5  FM  F`@ 
<
Q(0:/@@ 5  FE   MM  F`@@ ₯   M
<
K(0:/@@ 5  FE   MM  F`@@ ₯   M
<
V(0:/@@ 5  FE   MM  F`@@ ₯   M
8
K_cache(0J%@ @5  FM  M`@@ ₯   M
8
V_cache(0J%@ @5  FM  M`@@ ₯   M
?

MHA_GEMM_1(0B)@ @5  FE  MM  J`@@ 
2
SOFTMAX	(0J @5  JM  J`@ 
4
	DropOut_1
(0J @5  JM  J`@ 
?

MHA_GEMM_2(0B)@@ 5  ME  JM  F`@ΦJ 
K
	PROJ_GEMM(0:6@@ 5  FE   MM  F`hu  F@@ ₯   M
2
	DropOut_2	(0J@ 5  FM  F`@ 
4
Add_1
(0R#@ 5  FM  F`@ υ  F
4
LayerNorm_2(0J@ 5  FM  F`@ 
A
FFN0(0:1ΰ@ 5  FE  ΰMM  `G`ΰ@ ₯  ΰM
/
GeLU(0Jΰ 5  `GM  `G`ΰ 
H
FFN1(0:8@ΰ 5  `GE  ΰMM  F`hu  F@ΰ ₯  ΰM
2
	DropOut_3(0J@ 5  FM  F`@ 
4
Add_2(0R#@ 5  FM  F`@ υ  F7-   0=  FE ΰMPZAdd_Prev_LayerbLayerNorm_1 %0=  FE  FPZLayerNorm_1bQ %0=  FE  FPZLayerNorm_1bK %0=  FE  FPZLayerNorm_1bV !0=  FE  FPZKbK_cache !0=  FE  FPZVbV_cache $0=  FE  FPZQb
MHA_GEMM_1 *0=  ME  MPZK_cacheb
MHA_GEMM_1 *	0=  JE  JP	Z
MHA_GEMM_1bSOFTMAX )	
0=  JE  JP
ZSOFTMAXb	DropOut_1 ,
0=  JE  JPZ	DropOut_1b
MHA_GEMM_2 *0=  ME  MPZV_cacheb
MHA_GEMM_2 ,0=  FE  FPZ
MHA_GEMM_2b	PROJ_GEMM 0-   0=  FE  FPZ	PROJ_GEMMb	DropOut_2 ,-   0=  FE  FPZ	DropOut_2bAdd_1 .-   0=  FE  FPZAdd_1bLayerNorm_2 1-   0=  FE ΰMPZAdd_Prev_LayerbAdd_1 (0=  FE  FPZLayerNorm_2bFFN0 &-   0=  `GE  `GPZFFN0bGeLU &-   0=  `GE  `GPZGeLUbFFN1 +-   0=  FE  FPZFFN1b	DropOut_3 ,-   0=  FE  FPZ	DropOut_3bAdd_2 (-   0=  FE  FPZAdd_1bAdd_2 3 χ6 -ΝΜ$QUΝΜ?" ­ffΦK²PP’
qKΜΜΜQ(   @  ΐA  ?% θ G-Γd*<EτύT=M(a&>U ;D":@@ (PXd`xpx*o:΄