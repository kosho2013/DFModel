
Ã)
7
Add_Prev_Layer(0JP 5   LM   L`P 
8
LayerNorm_1 (0JP 5   LM   L`
 
:
Q (0:+PP 5   LE  HMM   L`
P 
:
K (0:+PP 5   LE  HMM   L`
P 
:
V (0:+PP 5   LE  HMM   L`
P 
D

MHA_GEMM_1 (0B,P 5   LE   LM   N`  
5
SOFTMAX (0J P 5   NM   N`  
7
	DropOut_1 (0J P 5   NM   N`  
C

MHA_GEMM_2	 (0B+P 5   LE   NM   L`
 
I
	PROJ_GEMM
 	(0:2PP 5   LE  HMM   L`hu   LP
 
6
	DropOut_2 
(0JP 5   LM   L`P 
8
Add_1	 (0R%P 5   LM   L`P õ   L
8
LayerNorm_2
 (0JP 5   LM   L`P 
C
FFN0 (0:1ÀP 5   LE  HNM   M`u   (P 
2
GeLU (0J À 5   MM   M`( 
E
FFN1 (0:3PÀ 5   ME  HNM   L`hu   LP( 
6
	DropOut_3 (0JP 5   LM   L`
 
8
Add_2 (0R%P 5   LM   L`
 õ   L
3
Loss_bwd (0JP 5   LM   L`P 
:
DropOut_3_bwd (0JP 5   LM   L`P 
B
FFN1_bwd (0:,ÀP 5   LE  HNM   M`(P 
6
GeLU_bwd (0J À 5   MM   M`( 
I
FFN0_bwd (0:3PÀ 5   ME  HNM   L`hu   LP( 
<
LayerNorm_2_bwd (0JP 5   LM   L`P 
:
DropOut_2_bwd (0JP 5   LM   L`P 
F
PROJ_GEMM_bwd (0:+PP 5   LE  HMM   L`
P 
I
MHA_GEMM_2_bwd1 (0B,P 5   LE   LM   N`  
H
MHA_GEMM_2_bwd2 (0B+P 5   LE   NM   L`
 
E
V_bwd	 (0:2PP 5   LE  HMM   L`hu   LP
 
;
DropOut_1_bwd	 (0J P 5   NM   N`  
9
SOFTMAX_bwd
 (0J P 5   NM   N`  
H
MHA_GEMM_1_bwd1  (0B+P 5   NE   LM   L`
 
H
MHA_GEMM_1_bwd2!  (0B+P 5   NE   LM   L`
 
>
Q_bwd" !(0:+PP 5   LE  HMM   L`
P 
>
K_bwd# "(0:+PP 5   LE  HMM   L`
P 
W
FFN1_bwd_weight_update$ #(0B3À P5   LE   MM  HN`hu  HN(P 
X
FFN0_bwd_weight_update% $(0B4P À5   ME   LM  HN`hu  HN
À 
[
PROJ_GEMM_bwd_weight_update& %(0B2P P5   LE   LM  HM`hu  HM
P 
S
V_bwd_weight_update'	 &(0B2P P5   LE   LM  HM`hu  HM
P 
S
K_bwd_weight_update( '(0B2P P5   LE   LM  HM`hu  HM
P 
S
Q_bwd_weight_update) ((0B2P P5   LE   LM  HM`hu  HM
P 20=   LE   JPZAdd_Prev_LayerbLayerNorm_1 %0=   LE   IPZLayerNorm_1bQ %0=   LE   IPZLayerNorm_1bK %0=   LE   IPZLayerNorm_1bV $0=   LE   IPZQb
MHA_GEMM_1 $0=   LE   IPZKb
MHA_GEMM_1 *0=   NE   KPZ
MHA_GEMM_1bSOFTMAX )0=   NE   KPZSOFTMAXb	DropOut_1 $	0=   LE   IP	ZVb
MHA_GEMM_2 ,	0=   NE   KP
Z	DropOut_1b
MHA_GEMM_2 ,	
0=   LE   IPZ
MHA_GEMM_2b	PROJ_GEMM +
0=   LE   JPZ	PROJ_GEMMb	DropOut_2 '0=   LE   JPZ	DropOut_2bAdd_1 ,0
=   LE   JPZAdd_Prev_LayerbAdd_1 )0=   LE   JPZAdd_1bLayerNorm_2 (0=   LE   JPZLayerNorm_2bFFN0 !0=   ME   JPZFFN0bGeLU !0=   ME   JPZGeLUbFFN1 &0=   LE   JPZFFN1b	DropOut_3 '0=   LE   IPZ	DropOut_3bAdd_2 #0=   LE   JPZAdd_1bAdd_2 .0=   LE   JPZLoss_bwdbDropOut_3_bwd .0=   LE   JPZDropOut_3_bwdbFFN1_bwd )0=   ME   JPZFFN1_bwdbGeLU_bwd )0=   ME   JPZGeLU_bwdbFFN0_bwd 00=   LE   JPZFFN0_bwdbLayerNorm_2_bwd 50=   LE   JPZLayerNorm_2_bwdbDropOut_2_bwd 30=   LE   JPZDropOut_2_bwdbPROJ_GEMM_bwd 50=   LE   IPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1 )0=   LE   IPZVbMHA_GEMM_2_bwd1 50=   LE   IPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2 10=   NE   KP Z	DropOut_1bMHA_GEMM_2_bwd2 50=   NE   KP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd -0=   LE   IP"ZMHA_GEMM_2_bwd2bV_bwd 10=   NE   KP#ZDropOut_1_bwdbSOFTMAX_bwd 3 0=   NE   KP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1 3!0=   NE   KP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2 ) 0
=   LE   IP&ZKbMHA_GEMM_1_bwd1 )!0
=   LE   IP'ZQbMHA_GEMM_1_bwd2 - "0=   LE   IP(ZMHA_GEMM_1_bwd1bQ_bwd -!#0=   LE   IP)ZMHA_GEMM_1_bwd2bK_bwd <$0=   LE   JP*ZDropOut_3_bwdbFFN1_bwd_weight_update 3$0=   ME   JP+ZGeLUbFFN1_bwd_weight_update 7%0	=   ME   JP,ZGeLU_bwdbFFN0_bwd_weight_update :%0=   LE   JP-ZLayerNorm_2bFFN0_bwd_weight_update A&0=   LE   JP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update >	&0=   LE   IP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update ;'0=   LE   IP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update 7'0	=   LE   IP1ZLayerNorm_1bV_bwd_weight_update ;!(0=   LE   IP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update 7(0=   LE   IP3ZLayerNorm_1bK_bwd_weight_update ; )0=   LE   IP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update 7)0=   LE   IP5ZLayerNorm_1bQ_bwd_weight_update O° -  ¨LUáz´?b* ¨° ½  CÅ  HAÍ  HAÒTPÚPPâDP¢
 àþD   Q-   @  ÀA  ?% è G-Ãd*<5ôýT=EôýT=M(a&>U ;D"!:PP (<X`xp *o:´