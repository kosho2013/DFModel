
É&
;
Add_Prev_Layer ˙˙˙˙˙˙˙˙˙(0J( 5   MM   M` 
:
LayerNorm_1 ˙˙˙˙˙˙˙˙˙(0J( 5   MM   M` 
8
Q ˙˙˙˙˙˙˙˙˙(0: (( 5   ME  HLM   M` 
8
K ˙˙˙˙˙˙˙˙˙(0: (( 5   ME  HLM   M` 
8
V ˙˙˙˙˙˙˙˙˙(0: (( 5   ME  HLM   M` 
B

MHA_GEMM_1 ˙˙˙˙˙˙˙˙˙(0B!( 5   ME   MM   P` 
7
SOFTMAX ˙˙˙˙˙˙˙˙˙(0J( 5   PM   P` 
9
	DropOut_1 ˙˙˙˙˙˙˙˙˙(0J( 5   PM   P` 
B

MHA_GEMM_2	 ˙˙˙˙˙˙˙˙˙(0B!( 5   ME   PM   M` 
G
	PROJ_GEMM
 ˙˙˙˙˙˙˙˙˙(0:'(( 5   ME  HLM   M`hu   M 
8
	DropOut_2 ˙˙˙˙˙˙˙˙˙(0J( 5   MM   M` 
:
Add_1	 ˙˙˙˙˙˙˙˙˙(0R( 5   MM   M` ő   M
:
LayerNorm_2
 ˙˙˙˙˙˙˙˙˙(0J( 5   MM   M` 
A
FFN0 ˙˙˙˙˙˙˙˙˙(0:& ( 5   ME  HMM   N`u    
4
GeLU ˙˙˙˙˙˙˙˙˙(0J  5   NM   N` 
C
FFN1 ˙˙˙˙˙˙˙˙˙(0:((  5   NE  HMM   M`hu   M 
8
	DropOut_3 ˙˙˙˙˙˙˙˙˙(0J( 5   MM   M` 
:
Add_2 ˙˙˙˙˙˙˙˙˙(0R( 5   MM   M` ő   M
5
Loss_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   MM   M` 
<
DropOut_3_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   MM   M` 
@
FFN1_bwd ˙˙˙˙˙˙˙˙˙(0:! ( 5   ME  HMM   N` 
8
GeLU_bwd ˙˙˙˙˙˙˙˙˙(0J  5   NM   N` 
G
FFN0_bwd ˙˙˙˙˙˙˙˙˙(0:((  5   NE  HMM   M`hu   M 
>
LayerNorm_2_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   MM   M` 
<
DropOut_2_bwd ˙˙˙˙˙˙˙˙˙(0J( 5   MM   M` 
D
PROJ_GEMM_bwd ˙˙˙˙˙˙˙˙˙(0: (( 5   ME  HLM   M` 
G
MHA_GEMM_2_bwd1 ˙˙˙˙˙˙˙˙˙(0B!( 5   ME   MM   P` 
G
MHA_GEMM_2_bwd2 ˙˙˙˙˙˙˙˙˙(0B!( 5   ME   PM   M` 
C
V_bwd	 ˙˙˙˙˙˙˙˙˙(0:'(( 5   ME  HLM   M`hu   M 
=
DropOut_1_bwd	 ˙˙˙˙˙˙˙˙˙(0J( 5   PM   P` 
;
SOFTMAX_bwd
 ˙˙˙˙˙˙˙˙˙(0J( 5   PM   P` 
G
MHA_GEMM_1_bwd1  ˙˙˙˙˙˙˙˙˙(0B!( 5   PE   MM   M` 
G
MHA_GEMM_1_bwd2! ˙˙˙˙˙˙˙˙˙(0B!( 5   PE   MM   M` 
<
Q_bwd" ˙˙˙˙˙˙˙˙˙(0: (( 5   ME  HLM   M` 
<
K_bwd# ˙˙˙˙˙˙˙˙˙(0: (( 5   ME  HLM   M` 
U
FFN1_bwd_weight_update$ ˙˙˙˙˙˙˙˙˙(0B(  (5   ME   NM  HM`hu  HM 
U
FFN0_bwd_weight_update% ˙˙˙˙˙˙˙˙˙(0B((  5   NE   MM  HM`hu  HM 
Y
PROJ_GEMM_bwd_weight_update& ˙˙˙˙˙˙˙˙˙(0B'( (5   ME   MM  HL`hu  HL 
Q
V_bwd_weight_update'	 ˙˙˙˙˙˙˙˙˙(0B'( (5   ME   MM  HL`hu  HL 
Q
K_bwd_weight_update( ˙˙˙˙˙˙˙˙˙(0B'( (5   ME   MM  HL`hu  HL 
Q
Q_bwd_weight_update) ˙˙˙˙˙˙˙˙˙(0B'( (5   ME   MM  HL`hu  HL /-   0=   MPZAdd_Prev_LayerbLayerNorm_10=   MPZLayerNorm_1bQ0=   MPZLayerNorm_1bK0=   MPZLayerNorm_1bV0=   MPZQb
MHA_GEMM_10=   MPZKb
MHA_GEMM_1"0=   PPZ
MHA_GEMM_1bSOFTMAX!0=   PPZSOFTMAXb	DropOut_1	0=   MP	ZVb
MHA_GEMM_2$	0=   PP
Z	DropOut_1b
MHA_GEMM_2$	
0=   MPZ
MHA_GEMM_2b	PROJ_GEMM(
-   0=   MPZ	PROJ_GEMMb	DropOut_2$-   0=   MPZ	DropOut_2bAdd_1)-   0
=   MPZAdd_Prev_LayerbAdd_1&-   0=   MPZAdd_1bLayerNorm_2 0=   MPZLayerNorm_2bFFN0-   0=   NPZFFN0bGeLU0=   NPZGeLUbFFN1#-   0=   MPZFFN1b	DropOut_30=   MPZ	DropOut_3bAdd_20=   MPZAdd_1bAdd_2&0=   MPZLoss_bwdbDropOut_3_bwd&0=   MPZDropOut_3_bwdbFFN1_bwd&-   0=   NPZFFN1_bwdbGeLU_bwd!0=   NPZGeLU_bwdbFFN0_bwd--   0=   MPZFFN0_bwdbLayerNorm_2_bwd2-   0=   MPZLayerNorm_2_bwdbDropOut_2_bwd+0=   MPZDropOut_2_bwdbPROJ_GEMM_bwd-0=   MPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1!0=   MPZVbMHA_GEMM_2_bwd1-0=   MPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2)0=   PP Z	DropOut_1bMHA_GEMM_2_bwd2-0=   PP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd%0=   MP"ZMHA_GEMM_2_bwd2bV_bwd)0=   PP#ZDropOut_1_bwdbSOFTMAX_bwd+ 0=   PP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1+!0=   PP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2! 0
=   MP&ZKbMHA_GEMM_1_bwd1!!0
=   MP'ZQbMHA_GEMM_1_bwd2% "0=   MP(ZMHA_GEMM_1_bwd1bQ_bwd%!#0=   MP)ZMHA_GEMM_1_bwd2bK_bwd4$0=   MP*ZDropOut_3_bwdbFFN1_bwd_weight_update+$0=   NP+ZGeLUbFFN1_bwd_weight_update/%0	=   NP,ZGeLU_bwdbFFN0_bwd_weight_update2%0=   MP-ZLayerNorm_2bFFN0_bwd_weight_update9&0=   MP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update6	&0=   MP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update3'0=   MP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update4'-   0	=   MP1ZLayerNorm_1bV_bwd_weight_update3!(0=   MP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update/(0=   MP3ZLayerNorm_1bK_bwd_weight_update3 )0=   MP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update/)0=   MP5ZLayerNorm_1bQ_bwd_weight_update":  -   M5   ?=  ČBE  ŔS˘    @  ŔA  ?"!:(( ((X hpx 