
΅
/
Add_Prev_Layer(0J@ 5   LM   L` 
.
LayerNorm_1(0J@ 5   LM   L` 
,
Q(0:@@ 5   LE   MM   L` 
,
K(0:@@ 5   LE   MM   L` 
,
V(0:@@ 5   LE   MM   L` 
5

MHA_GEMM_1(0B@ 5   LE   LM   N` 
*
SOFTMAX(0J@ 5   NM   N` 
,
	DropOut_1(0J@ 5   NM   N` 
5

MHA_GEMM_2	(0B@ 5   LE   NM   L` 
;
	PROJ_GEMM
(0:&@@ 5   LE   MM   L`hu   L 
,
	DropOut_2(0J@ 5   LM   L` 
0
Add_1	 (0R@ 5   LM   L` υ   L
0
LayerNorm_2
 (0J@ 5   LM   L` 
7
FFN0 (0:%@ 5   LE   NM   M`u    
*
GeLU (0J 5   MM   M` 
9
FFN1 (0:'@ 5   ME   NM   L`hu   L 
.
	DropOut_3 (0J@ 5   LM   L` 
0
Add_2 (0R@ 5   LM   L` υ   L2-   0=   LPZAdd_Prev_LayerbLayerNorm_1  0=   LPZLayerNorm_1bQ  0=   LPZLayerNorm_1bK  0=   LPZLayerNorm_1bV 0=   LPZQb
MHA_GEMM_1 0=   LPZKb
MHA_GEMM_1 %0=   NPZ
MHA_GEMM_1bSOFTMAX $0=   NPZSOFTMAXb	DropOut_1 	0=   LP	ZVb
MHA_GEMM_2 '	0=   NP
Z	DropOut_1b
MHA_GEMM_2 '	
0=   LPZ
MHA_GEMM_2b	PROJ_GEMM +
-   0=   LPZ	PROJ_GEMMb	DropOut_2 '-   0=   LPZ	DropOut_2bAdd_1 ,-   0
=   LPZAdd_Prev_LayerbAdd_1 )-   0=   LPZAdd_1bLayerNorm_2 (-   0=   LPZLayerNorm_2bFFN0 !-   0=   MPZFFN0bGeLU !-   0=   MPZGeLUbFFN1 &-   0=   LPZFFN1b	DropOut_3 '-   0=   LPZ	DropOut_3bAdd_2 #-   0=   LPZAdd_1bAdd_2 X)  -  N0=   I@(H4UΝΜΜ?X browwiseZ ¨΅   A½   AΒTPΚPP’
 ΐΜD  Q(   @  ΐA  ?%F-τύT=EτύT=M(a&>UeZήC"":@@ (0X`h xpx*o: