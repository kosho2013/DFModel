
�&
;
Add_Prev_Layer ���������(0J�� �5  �MM  �M`�
:
LayerNorm_1 ���������(0J�� �5  �MM  �M`�
9
Q ���������(0:!���� �5  �ME @�PM  �M`�
9
K ���������(0:!���� �5  �ME @�PM  �M`�
9
V ���������(0:!���� �5  �ME @�PM  �M`�
A

MHA_GEMM_1 ���������(0B ��� �5  �ME  �MM   O`�
6
SOFTMAX ���������(0J�� �5   OM   O`�
8
	DropOut_1 ���������(0J�� �5   OM   O`�
A

MHA_GEMM_2	 ���������(0B ��� �5  �ME   OM  �M`�
H
	PROJ_GEMM
 ���������(0:(���� �5  �ME @�PM  �M`hu  �M�
8
	DropOut_2 ���������(0J�� �5  �MM  �M`�
:
Add_1	 ���������(0R�� �5  �MM  �M`��  �M
:
LayerNorm_2
 ���������(0J�� �5  �MM  �M`�
A
FFN0 ���������(0:&���� �5  �ME @�QM  �N`u   ��
3
GeLU ���������(0J�� �5  �NM  �N`�
C
FFN1 ���������(0:(���� �5  �NE @�QM  �M`hu  �M�
8
	DropOut_3 ���������(0J�� �5  �MM  �M`�
:
Add_2 ���������(0R�� �5  �MM  �M`��  �M
5
Loss_bwd ���������(0J�� �5  �MM  �M`�
<
DropOut_3_bwd ���������(0J�� �5  �MM  �M`�
@
FFN1_bwd ���������(0:!���� �5  �ME @�QM  �N`�
7
GeLU_bwd ���������(0J�� �5  �NM  �N`�
G
FFN0_bwd ���������(0:(���� �5  �NE @�QM  �M`hu  �M�
>
LayerNorm_2_bwd ���������(0J�� �5  �MM  �M`�
<
DropOut_2_bwd ���������(0J�� �5  �MM  �M`�
E
PROJ_GEMM_bwd ���������(0:!���� �5  �ME @�PM  �M`�
F
MHA_GEMM_2_bwd1 ���������(0B ��� �5  �ME  �MM   O`�
F
MHA_GEMM_2_bwd2 ���������(0B ��� �5  �ME   OM  �M`�
D
V_bwd	 ���������(0:(���� �5  �ME @�PM  �M`hu  �M�
<
DropOut_1_bwd	 ���������(0J�� �5   OM   O`�
:
SOFTMAX_bwd
 ���������(0J�� �5   OM   O`�
F
MHA_GEMM_1_bwd1  ���������(0B ��� �5   OE  �MM  �M`�
F
MHA_GEMM_1_bwd2! ���������(0B ��� �5   OE  �MM  �M`�
=
Q_bwd" ���������(0:!���� �5  �ME @�PM  �M`�
=
K_bwd# ���������(0:!���� �5  �ME @�PM  �M`�
U
FFN1_bwd_weight_update$ ���������(0B(��� ��5  �ME  �NM @�Q`hu @�Q�
U
FFN0_bwd_weight_update% ���������(0B(��� ��5  �NE  �MM @�Q`hu @�Q�
Z
PROJ_GEMM_bwd_weight_update& ���������(0B(��� ��5  �ME  �MM @�P`hu @�P�
R
V_bwd_weight_update'	 ���������(0B(��� ��5  �ME  �MM @�P`hu @�P�
R
K_bwd_weight_update( ���������(0B(��� ��5  �ME  �MM @�P`hu @�P�
R
Q_bwd_weight_update) ���������(0B(��� ��5  �ME  �MM @�P`hu @�P�/-   �0=  �MPZAdd_Prev_LayerbLayerNorm_10=  �MPZLayerNorm_1bQ0=  �MPZLayerNorm_1bK0=  �MPZLayerNorm_1bV0=  �MPZQb
MHA_GEMM_10=  �MPZKb
MHA_GEMM_1"0=   OPZ
MHA_GEMM_1bSOFTMAX!0=   OPZSOFTMAXb	DropOut_1	0=  �MP	ZVb
MHA_GEMM_2$	0=   OP
Z	DropOut_1b
MHA_GEMM_2$	
0=  �MPZ
MHA_GEMM_2b	PROJ_GEMM(
-   �0=  �MPZ	PROJ_GEMMb	DropOut_2$-   �0=  �MPZ	DropOut_2bAdd_1)-   �0
=  �MPZAdd_Prev_LayerbAdd_1&-   �0=  �MPZAdd_1bLayerNorm_2 0=  �MPZLayerNorm_2bFFN0-   �0=  �NPZFFN0bGeLU0=  �NPZGeLUbFFN1#-   �0=  �MPZFFN1b	DropOut_30=  �MPZ	DropOut_3bAdd_20=  �MPZAdd_1bAdd_2&0=  �MPZLoss_bwdbDropOut_3_bwd&0=  �MPZDropOut_3_bwdbFFN1_bwd&-   �0=  �NPZFFN1_bwdbGeLU_bwd!0=  �NPZGeLU_bwdbFFN0_bwd--   �0=  �MPZFFN0_bwdbLayerNorm_2_bwd2-   �0=  �MPZLayerNorm_2_bwdbDropOut_2_bwd+0=  �MPZDropOut_2_bwdbPROJ_GEMM_bwd-0=  �MPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1!0=  �MPZVbMHA_GEMM_2_bwd1-0=  �MPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2)0=   OP Z	DropOut_1bMHA_GEMM_2_bwd2-0=   OP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd%0=  �MP"ZMHA_GEMM_2_bwd2bV_bwd)0=   OP#ZDropOut_1_bwdbSOFTMAX_bwd+ 0=   OP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1+!0=   OP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2! 0
=  �MP&ZKbMHA_GEMM_1_bwd1!!0
=  �MP'ZQbMHA_GEMM_1_bwd2% "0=  �MP(ZMHA_GEMM_1_bwd1bQ_bwd%!#0=  �MP)ZMHA_GEMM_1_bwd2bK_bwd4$0=  �MP*ZDropOut_3_bwdbFFN1_bwd_weight_update+$0=  �NP+ZGeLUbFFN1_bwd_weight_update/%0	=  �NP,ZGeLU_bwdbFFN0_bwd_weight_update2%0=  �MP-ZLayerNorm_2bFFN0_bwd_weight_update9&0=  �MP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update6	&0=  �MP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update3'0=  �MP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update4'-   �0	=  �MP1ZLayerNorm_1bV_bwd_weight_update3!(0=  �MP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update/(0=  �MP3ZLayerNorm_1bK_bwd_weight_update3 )0=  �MP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update/)0=  �MP5ZLayerNorm_1bQ_bwd_weight_update7�:� -  �L5�"�?= �;EE  �Qb%  aD-  HB5  HB@
H@P   @  �A  �?"$:���� �(�0X�@px���