
®
6
Add_Prev_Layer(0J   5   LM   L`  
5
LayerNorm_1(0J   5   LM   L`  
7
Q(0:*    5   LE   LM   L`  
7
K(0:*    5   LE   LM   L`  
7
V(0:*    5   LE   LM   L`  
A

MHA_GEMM_1(0B+    5   LE   LM  N` 
2
SOFTMAX(0J    5  NM  N` 
4
	DropOut_1(0J    5  NM  N` 
@

MHA_GEMM_2	(0B*    5   LE  NM   L`  
F
	PROJ_GEMM
(0:1    5   LE   LM   L`hu   L  
3
	DropOut_2(0J   5   LM   L`  
5
Add_1	(0R$   5   LM   L`  õ   L
5
LayerNorm_2
(0J   5   LM   L`  
?
FFN0(0:/p   5   LE  àLM  àL`u     
.
GeLU(0Jp  5  àLM  àL` 
A
FFN1(0:1 p  5  àLE  àLM   L`hu   L  
3
	DropOut_3(0J   5   LM   L`  
5
Add_2(0R$   5   LM   L`  õ   L7-   0=   LE   FPZAdd_Prev_LayerbLayerNorm_1 %0=   LE   FPZLayerNorm_1bQ %0=   LE   FPZLayerNorm_1bK %0=   LE   FPZLayerNorm_1bV $0=   LE  DPZQb
MHA_GEMM_1 $0=   LE  DPZKb
MHA_GEMM_1 *0=  NE   GPZ
MHA_GEMM_1bSOFTMAX )0=  NE   GPZSOFTMAXb	DropOut_1 $	0=   LE  DP	ZVb
MHA_GEMM_2 ,	0=  NE   GP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=   LE  DPZ
MHA_GEMM_2b	PROJ_GEMM 0
-   0=   LE   FPZ	PROJ_GEMMb	DropOut_2 ,-   0=   LE   FPZ	DropOut_2bAdd_1 1-   0
=   LE   FPZAdd_Prev_LayerbAdd_1 .-   0=   LE   FPZAdd_1bLayerNorm_2 --   0=   LE   FPZLayerNorm_2bFFN0 &-   0=  àLE  `EPZFFN0bGeLU &-   0=  àLE  `EPZGeLUbFFN1 +-   0=   LE   FPZFFN1b	DropOut_3 ,-   0=   LE   FPZ	DropOut_3bAdd_2 (-   0=   LE   FPZAdd_1bAdd_2 F  -  NUÍÌÌ?½E  C: ¨µ  HB½  HBÂTPÊPP¢
ÍÌÌD  Q(   @  ÀA  ?% è G-Ãd*<EôýT=M(a&>U ;D"*:    ( X`xpxfff? *o:´