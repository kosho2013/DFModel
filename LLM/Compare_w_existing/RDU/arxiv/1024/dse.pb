
ç
8
Add_Prev_Layer ˙˙˙˙˙˙˙˙˙(0J( 5   KM   K 
5
LayerNorm_1 ˙˙˙˙˙˙˙˙˙(0J( 5   KM   K 
3
Q ˙˙˙˙˙˙˙˙˙(0:(( 5   KE  HLM   K 
3
K ˙˙˙˙˙˙˙˙˙(0:(( 5   KE  HLM   K 
3
V ˙˙˙˙˙˙˙˙˙(0:(( 5   KE  HLM   K 
<

MHA_GEMM_1 ˙˙˙˙˙˙˙˙˙(0B( 5   KE   KM   L 
1
SOFTMAX ˙˙˙˙˙˙˙˙˙(0J( 5   LM   L 
3
	DropOut_1 ˙˙˙˙˙˙˙˙˙(0J( 5   LM   L 
<

MHA_GEMM_2	 ˙˙˙˙˙˙˙˙˙(0B( 5   KE   LM   K 
;
	PROJ_GEMM
 ˙˙˙˙˙˙˙˙˙(0:(( 5   KE  HLM   K 
3
	DropOut_2 ˙˙˙˙˙˙˙˙˙(0J( 5   KM   K 
5
Add_1 ˙˙˙˙˙˙˙˙˙(0R( 5   KM   K ő   K
5
LayerNorm_2 ˙˙˙˙˙˙˙˙˙(0J( 5   KM   K 
7
FFN0 ˙˙˙˙˙˙˙˙˙(0: ( 5   KE  HMM   L 
/
GeLU ˙˙˙˙˙˙˙˙˙(0J  5   LM   L 
7
FFN1 ˙˙˙˙˙˙˙˙˙(0:(  5   LE  HMM   K 
3
	DropOut_3 ˙˙˙˙˙˙˙˙˙(0J( 5   KM   K 
5
Add_2 ˙˙˙˙˙˙˙˙˙(0R( 5   KM   K ő   K
2
Loss_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   KM   K 
7
DropOut_3_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   KM   K 
;
FFN1_bwd ˙˙˙˙˙˙˙˙˙(0: ( 5   KE  HMM   L 
3
GeLU_bwd ˙˙˙˙˙˙˙˙˙(0J  5   LM   L 
;
FFN0_bwd ˙˙˙˙˙˙˙˙˙(0:(  5   LE  HMM   K 
9
LayerNorm_2_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   KM   K 
7
DropOut_2_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   KM   K 
?
PROJ_GEMM_bwd ˙˙˙˙˙˙˙˙˙(0:(( 5   KE  HLM   K 
A
MHA_GEMM_2_bwd1 ˙˙˙˙˙˙˙˙˙(0B( 5   KE   KM   L 
A
MHA_GEMM_2_bwd2 ˙˙˙˙˙˙˙˙˙(0B( 5   KE   LM   K 
7
V_bwd ˙˙˙˙˙˙˙˙˙(0:(( 5   KE  HLM   K 
7
DropOut_1_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   LM   L 
5
SOFTMAX_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   LM   L 
A
MHA_GEMM_1_bwd1  ˙˙˙˙˙˙˙˙˙(0B( 5   LE   KM   K 
A
MHA_GEMM_1_bwd2! ˙˙˙˙˙˙˙˙˙(0B( 5   LE   KM   K 
7
Q_bwd" ˙˙˙˙˙˙˙˙˙(0:(( 5   KE  HLM   K 
7
K_bwd# ˙˙˙˙˙˙˙˙˙(0:(( 5   KE  HLM   K 
P
FFN1_bwd_weight_update$ ˙˙˙˙˙˙˙˙˙(0B%  (5   KE   LM  HMhu  HM 
P
FFN0_bwd_weight_update% ˙˙˙˙˙˙˙˙˙(0B%(  5   LE   KM  HMhu  HM 
T
PROJ_GEMM_bwd_weight_update& ˙˙˙˙˙˙˙˙˙(0B$( (5   KE   KM  HLhu  HL 
L
V_bwd_weight_update' ˙˙˙˙˙˙˙˙˙(0B$( (5   KE   KM  HLhu  HL 
L
K_bwd_weight_update( ˙˙˙˙˙˙˙˙˙(0B$( (5   KE   KM  HLhu  HL 
L
Q_bwd_weight_update) ˙˙˙˙˙˙˙˙˙(0B$( (5   KE   KM  HLhu  HL PPPPPPPP	P		P
	
P
PPPPPPPPPPPPPPPPPPPPP P!P"P# P$!P% P&!P' "P(!#P)$P*$P+%P,%P-&P.	&P/'P0'P1!(P2(P3 )P4)P5":  -   M5   ?=  ČBE  ŔS˘    @  ŔA  ?":(( ((X px 