
�)
6
Add_Prev_Layer(0J�( �5  �KM  �K`��(�#�
5
LayerNorm_1(0J�( �5  �KM  �K`��(�#�
7
Q(0:*(��( �5  �KE  HLM  �K`��(��:�#�
7
K(0:*(��( �5  �KE  HLM  �K`��(��'�#�
7
V(0:*(��( �5  �KE  HLM  �K`��(��+�#�
C

MHA_GEMM_1 (0B+(�� �5  �KE  �KM  �M`������#�
4
SOFTMAX (0J(� �5  �MM  �M`����Y�
6
	DropOut_1 (0J(� �5  �MM  �M`����#�
B

MHA_GEMM_2	 (0B*(�� �5  �KE  �MM  �K`��(���#�
H
	PROJ_GEMM
 (0:1�(�( �5  �KE  HLM  �K`hu  �K��(��(�#�
5
	DropOut_2 (0J�( �5  �KM  �K`��(�#�
7
Add_1	 (0R$�( �5  �KM  �K`��(�#��  �K
7
LayerNorm_2
 (0J�( �5  �KM  �K`��(�#�
C
FFN0 (0:1���( �5  �KE  HMM  �L`u   ������(�#�
2
GeLU (0J �� �5  �LM  �L`����#�
E
FFN1 (0:3�(�� �5  �LE  HMM  �K`hu  �K��=����#�
5
	DropOut_3 (0J�( �5  �KM  �K`��(�#�
7
Add_2 (0R$�( �5  �KM  �K`��(�#��  �K
3
Loss_bwd (0J�( �5  �KM  �K`��(��"�
9
DropOut_3_bwd (0J�( �5  �KM  �K`��(�#�
B
FFN1_bwd (0:,���( �5  �KE  HMM  �L`�����(�#�
6
GeLU_bwd (0J �� �5  �LM  �L`����#�
I
FFN0_bwd (0:3�(�� �5  �LE  HMM  �K`hu  �K��(����#�
;
LayerNorm_2_bwd (0J�( �5  �KM  �K`��(�#�
9
DropOut_2_bwd (0J�( �5  �KM  �K`��(�#�
E
PROJ_GEMM_bwd (0:*�(�( �5  �KE  HLM  �K`��(��(�#�
H
MHA_GEMM_2_bwd1 (0B+(�� �5  �KE  �KM  �M`������#�
G
MHA_GEMM_2_bwd2 (0B*(�� �5  �KE  �MM  �K`��(���#�
D
V_bwd	 (0:1�(�( �5  �KE  HLM  �K`hu  �K��(��(�#�
;
DropOut_1_bwd	 (0J (� �5  �MM  �M`������
8
SOFTMAX_bwd
 (0J(� �5  �MM  �M`����#�
G
MHA_GEMM_1_bwd1  (0B*(�� �5  �ME  �KM  �K`��(���#�
G
MHA_GEMM_1_bwd2! (0B*(�� �5  �ME  �KM  �K`��(���#�
=
Q_bwd" (0:*(��( �5  �KE  HLM  �K`��(��(�#�
=
K_bwd# (0:*(��( �5  �KE  HLM  �K`��'��(�#�
W
FFN1_bwd_weight_update$ (0B3��� �(5  �KE  �LM  HM`hu  HM����#��(�
W
FFN0_bwd_weight_update% (0B3�(� ��5  �LE  �KM  HM`hu  HM��(�#����
Z
PROJ_GEMM_bwd_weight_update& (0B1�(� �(5  �KE  �KM  HL`hu  HL��(�#��(�
R
V_bwd_weight_update'	 (0B1�(� �(5  �KE  �KM  HL`hu  HL��(�#��(�
R
K_bwd_weight_update( (0B1�(� �(5  �KE  �KM  HL`hu  HL��(�#��(�
R
Q_bwd_weight_update) (0B1�(� �(5  �KE  �KM  HL`hu  HL��(�#��(�20=  �KE  �HPZAdd_Prev_LayerbLayerNorm_1�%0=  �KE  �HPZLayerNorm_1bQ�%0=  �KE  �HPZLayerNorm_1bK�%0=  �KE  �HPZLayerNorm_1bV�$0=  �KE  �HPZQb
MHA_GEMM_1�$0=  �KE  �HPZKb
MHA_GEMM_1�*0=  �ME  �JPZ
MHA_GEMM_1bSOFTMAX�)0=  �ME�gbKPZSOFTMAXb	DropOut_1�$	0=  �KE  �HP	ZVb
MHA_GEMM_2�,	0=  �ME  �JP
Z	DropOut_1b
MHA_GEMM_2�,	
0=  �KE  �HPZ
MHA_GEMM_2b	PROJ_GEMM�+
0=  �KE  �HPZ	PROJ_GEMMb	DropOut_2�'0=  �KE  �HPZ	DropOut_2bAdd_1�,0
=  �KE  �HPZAdd_Prev_LayerbAdd_1�)0=  �KE  �HPZAdd_1bLayerNorm_2�(0=  �KE  �HPZLayerNorm_2bFFN0�!0=  �LE  �IPZFFN0bGeLU�!0=  �LEx JPZGeLUbFFN1�&0=  �KE`�IPZFFN1b	DropOut_3�'0=  �KE  �HPZ	DropOut_3bAdd_2�#0=  �KE  �HPZAdd_1bAdd_2�.0=  �KE p-LPZLoss_bwdbDropOut_3_bwd�.0=  �KE  �HPZDropOut_3_bwdbFFN1_bwd�)0=  �LE���IPZFFN1_bwdbGeLU_bwd�)0=  �LE  �IPZGeLU_bwdbFFN0_bwd�00=  �KE  �HPZFFN0_bwdbLayerNorm_2_bwd�50=  �KE  �HPZLayerNorm_2_bwdbDropOut_2_bwd�30=  �KE  �HPZDropOut_2_bwdbPROJ_GEMM_bwd�50=  �KE  �HPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd1�)0=  �KE  �HPZVbMHA_GEMM_2_bwd1�50=  �KE  �HPZPROJ_GEMM_bwdbMHA_GEMM_2_bwd2�10=  �ME  �JP Z	DropOut_1bMHA_GEMM_2_bwd2�50=  �ME  �JP!ZMHA_GEMM_2_bwd1bDropOut_1_bwd�-0=  �KE  �HP"ZMHA_GEMM_2_bwd2bV_bwd�10=  �ME  kLP#ZDropOut_1_bwdbSOFTMAX_bwd�3 0=  �ME `�JP$ZSOFTMAX_bwdbMHA_GEMM_1_bwd1�3!0=  �ME `�JP%ZSOFTMAX_bwdbMHA_GEMM_1_bwd2�) 0
=  �KE  �HP&ZKbMHA_GEMM_1_bwd1�)!0
=  �KE  �HP'ZQbMHA_GEMM_1_bwd2�- "0=  �KE  �HP(ZMHA_GEMM_1_bwd1bQ_bwd�-!#0=  �KE  �HP)ZMHA_GEMM_1_bwd2bK_bwd�<$0=  �KE  �HP*ZDropOut_3_bwdbFFN1_bwd_weight_update�3$0=  �LEx JP+ZGeLUbFFN1_bwd_weight_update�7%0	=  �LE  �IP,ZGeLU_bwdbFFN0_bwd_weight_update�:%0=  �KE  �HP-ZLayerNorm_2bFFN0_bwd_weight_update�A&0=  �KE  �HP.ZDropOut_2_bwdbPROJ_GEMM_bwd_weight_update�>	&0=  �KE  �HP/Z
MHA_GEMM_2bPROJ_GEMM_bwd_weight_update�;'0=  �KE  �HP0ZMHA_GEMM_2_bwd2bV_bwd_weight_update�7'0	=  �KE  �HP1ZLayerNorm_1bV_bwd_weight_update�;!(0=  �KE  �HP2ZMHA_GEMM_1_bwd2bK_bwd_weight_update�7(0=  �KE  �HP3ZLayerNorm_1bK_bwd_weight_update�; )0=  �KE  �HP4ZMHA_GEMM_1_bwd1bQ_bwd_weight_update�7)0=  �KE  �HP5ZLayerNorm_1bQ_bwd_weight_update�-�  -  �MU  �?"��  HB�DP�  �B-   @  �A  �?% � G-�d*<5��T=E��T=M(a&>U �;D" :�(�( �((X�@`xpx��*o�:�