
�)
6
Add_Prev_Layer(0J�( �@5  �LM  �L`��d��
5
LayerNorm_1(0J�( �@5  �LM  �L`��(��
7
Q(0:*(��( �@5  �LE  HLM  �L`��(��(��
7
K(0:*(��( �@5  �LE  HLM  �L`��(��(��
7
V(0:*(��( �@5  �LE  HLM  �L`��(��(��
C

MHA_GEMM_1 (0B+(�@� �@5  �LE  �LM  �O`�������
4
SOFTMAX (0J(�@ �@5  �OM  �O`�����
6
	DropOut_1 (0J(�@ �@5  �OM  �O`�����
B

MHA_GEMM_2	 (0B*(��@ �@5  �LE  �OM  �L`��(��A��
H
	PROJ_GEMM
 (0:1�(�( �@5  �LE  HLM  �L`hu  �L��(��(��
5
	DropOut_2 (0J�( �@5  �LM  �L`��(��
7
Add_1	 (0R$�( �@5  �LM  �L`��(���  �L
7
LayerNorm_2
 (0J�( �@5  �LM  �L`��(��
C
FFN0 (0:1���( �@5  �LE  HMM  �M`u   ������(��
2
GeLU (0J �� �@5  �MM  �M`�����
E
FFN1 (0:3�(�� �@5  �ME  HMM  �L`hu  �L��(�����
5
	DropOut_3 (0J�( �@5  �LM  �L`��(��
7
Add_2 (0R$�( �@5  �LM  �L`��(���  �L
2
Loss_bwd (0J�( �@5  �LM  �L`��(��
9
DropOut_3_bwd (0J�( �@5  �LM  �L`��(��
B
FFN1_bwd (0:,���( �@5  �LE  HMM  �M`�����(��
6
GeLU_bwd (0J �� �@5  �MM  �M`�����
I
FFN0_bwd (0:3�(�� �@5  �ME  HMM  �L`hu  �L��(�����
<
LayerNorm_2_bwd (0J�( �@5  �LM  �L`��y���
9
DropOut_2_bwd (0J�( �@5  �LM  �L`��(��
E
PROJ_GEMM_bwd (0:*�(�( �@5  �LE  HLM  �L`��(��(��
H
MHA_GEMM_2_bwd1 (0B+(�@� �@5  �LE  �LM  �O`�������
G
MHA_GEMM_2_bwd2 (0B*(��@ �@5  �LE  �OM  �L`��(��@��
D
V_bwd	 (0:1�(�( �@5  �LE  HLM  �L`hu  �L��(��(��
:
DropOut_1_bwd	 (0J(�@ �@5  �OM  �O`�����
8
SOFTMAX_bwd
 (0J(�@ �@5  �OM  �O`�����
G
MHA_GEMM_1_bwd1  (0B*(��@ �@5  �OE  �LM  �L`��(��i��
G
MHA_GEMM_1_bwd2! (0B*(��@ �@5  �OE  �LM  �L`��(��@��
=
Q_bwd" (0:*(��( �@5  �LE  HLM  �L`��(��(��
=
K_bwd# (0:*(��( �@5  �LE  HLM  �L`��(��(��
W
FFN1_bwd_weight_update$ (0B3���@ �(5  �LE  �MM  HM`hu  HM������(�
W
FFN0_bwd_weight_update% (0B3�(�@ ��5  �ME  �LM  HM`hu  HM��(�����
Z
PROJ_GEMM_bwd_weight_update& (0B1�(�@ �(5  �LE  �LM  HL`hu  HL��(���(�
R
V_bwd_weight_update'	 (0B1�(�@ �(5  �LE  �LM  HL`hu  HL��(���(�
R
K_bwd_weight_update( (0B1�(�@ �(5  �LE  �LM  HL`hu  HL��(���(�
R
Q_bwd_weight_update) (0B1�(�@ �(5  �LE  �LM  HL`hu  HL��(���(�20=  �LE ��HPZAdd_Prev_LayerbLayerNorm_1�%0=  �LE  �HPZLayerNorm_1bQ�%0=  �LE  �HPZLayerNorm_1bK�%0=  �LE  �HPZLayerNorm_1bV�$0=  �LE  HPZQb
MHA_GEMM_1�$0=  �LE  HPZKb
MHA_GEMM_1�*0=  �OE  KPZ
MHA_GEMM_1bSOFTMAX�)0=  �OE  KPZSOFTMAXb	DropOut_1�$	0=  �LE  HP	ZVb
MHA_GEMM_2�,	0=  �OE  KP
Z	DropOut_1b
MHA_GEMM_2�,	
0=  �LE  HPZ
MHA_GEMM_2b	PROJ_GEMM�+
0=  �LE  HPZ	PROJ_GEMMb	DropOut_2�'0=  �LE  HPZ	DropOut_2bAdd_1�,0
=  �LE ��HPZAdd_Prev_LayerbAdd_1�)0=  �LE  HPZAdd_1bLayerNorm_2�(0=  �LE  HPZLayerNorm_2bFFN0�!0=  �ME  IPZFFN0bGeLU�!0=  �ME   IPZGeLUbFFN1�&0=  �LE  HPZFFN1b	DropOut_3�'0=  �LE  HPZ	DropOut_3bAdd_2�#0=  �LE  HPZAdd_1bAdd_2�.0=  �LE  HPZLoss_bwdbDropOut_3_bwd�.0=  �LE  HPZDropOut_3_bwdbFFN1_bwd�)0=  �ME  IPZFFN1_bwdbGeLU_bwd�)0=  �ME  IPZGeLU_bwdbFFN0_bwd�00=  �LE  HPZFFN0_bwdbLayerNorm_2_bwd�50=  �LE ��JPZLayerNorm_2_bwdbDropOut_2_bwd�30=  �LE  HPZDropOut_2_bwdbPROJ_GEMM_bwd�50=  �LE  HPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1�)0=  �LE  HPZVbMHA_GEMM_2_bwd1�50=  �LE  HPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2�10=  �OE  KP Z	DropOut_1bMHA_GEMM_2_bwd2�50=  �OE  KP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd�-0=  �LE  HP"ZMHA_GEMM_2_bwd2bV_bwd�10=  �OE  KP#ZDropOut_1_bwdbSOFTMAX_bwd�3 0=  �OE  KP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1�3!0=  �OE  KP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2�) 0
=  �LE  HP&ZKbMHA_GEMM_1_bwd1�)!0
=  �LE  HP'ZQbMHA_GEMM_1_bwd2�- "0=  �LE  HP(ZMHA_GEMM_1_bwd1bQ_bwd�-!#0=  �LE  HP)ZMHA_GEMM_1_bwd2bK_bwd�<$0=  �LE  HP*ZDropOut_3_bwdbFFN1_bwd_weight_update�3$0=  �ME   IP+ZGeLUbFFN1_bwd_weight_update�7%0	=  �ME  IP,ZGeLU_bwdbFFN0_bwd_weight_update�:%0=  �LE  HP-ZLayerNorm_2bFFN0_bwd_weight_update�A&0=  �LE  HP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update�>	&0=  �LE  HP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update�;'0=  �LE  HP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update�7'0	=  �LE  �HP1ZLayerNorm_1bV_bwd_weight_update�;!(0=  �LE  HP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update�7(0=  �LE  �HP3ZLayerNorm_1bK_bwd_weight_update�; )0=  �LE  HP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update�7)0=  �LE  �HP5ZLayerNorm_1bQ_bwd_weight_update�-�  -  �MU  �?"��  HB�DP�  �B-   @  �A  �?% � G-�d*<5��T=E��T=M(a&>U �;D" :�(�( �@((X�@`xpx��*o�:�