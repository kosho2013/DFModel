
�
7
Add_Prev_Layer(0J�` �u5 �/MM �/M`��`��u�
6
LayerNorm_1(0J�` �u5 �/MM �/M`��`��u�
8
Q(0:+`��` �u5 �/ME  MM �/M`����`��u�
8
K(0:+`��` �u5 �/ME  MM �/M`����`��u�
8
V(0:+`��` �u5 �/ME  MM �/M`����`��u�
B

MHA_GEMM_1(0B,`�u� �u5 �/ME �/MM��P`�������u�
3
SOFTMAX(0J `�u �u5��PM��P`�����u�
5
	DropOut_1(0J `�u �u5��PM��P`�����u�
A

MHA_GEMM_2	(0B+`��u �u5 �/ME��PM �/M`����u��u�
G
	PROJ_GEMM
(0:2�`�` �u5 �/ME  MM �/M`hu �/M��`����u�
4
	DropOut_2(0J�` �u5 �/MM �/M`��`��u�
6
Add_1	(0R%�` �u5 �/MM �/M`��`��u�� �/M
6
LayerNorm_2
(0J�` �u5 �/MM �/M`��`��u�
<
FFN0(0:,���` �u5 �/ME  NM �/N`�� ��`��u�
0
GeLU(0J �� �u5 �/NM �/N`�� ��u�
C
FFN1(0:3�`�� �u5 �/NE  NM �/M`hu �/M��`�� ��u�
4
	DropOut_3(0J�` �u5 �/MM �/M`��`��u�
6
Add_2(0R%�` �u5 �/MM �/M`��`��u�� �/M7-   �0= �/ME �/MPZAdd_Prev_LayerbLayerNorm_1�%0= �/ME �/MPZLayerNorm_1bQ�%0= �/ME �/MPZLayerNorm_1bK�%0= �/ME �/MPZLayerNorm_1bV�$0= �/ME `jKPZQb
MHA_GEMM_1�$0= �/ME `jKPZKb
MHA_GEMM_1�*0=��PE���NPZ
MHA_GEMM_1bSOFTMAX�)0=��PE���NPZSOFTMAXb	DropOut_1�$	0= �/ME `jKP	ZVb
MHA_GEMM_2�,	0=��PE���NP
Z	DropOut_1b
MHA_GEMM_2�,	
0= �/ME `jKPZ
MHA_GEMM_2b	PROJ_GEMM�0
-   �0= �/ME �/MPZ	PROJ_GEMMb	DropOut_2�,-   �0= �/ME �/MPZ	DropOut_2bAdd_1�1-   �0
= �/ME �/MPZAdd_Prev_LayerbAdd_1�.-   �0= �/ME �/MPZAdd_1bLayerNorm_2�--   �0= �/ME �/MPZLayerNorm_2bFFN0�&-   �0= �/NE `jLPZFFN0bGeLU�&-   �0= �/NE `jLPZGeLUbFFN1�+-   �0= �/ME �/MPZFFN1b	DropOut_3�,-   �0= �/ME �/MPZ	DropOut_3bAdd_2�(-   �0= �/ME �/MPZAdd_1bAdd_2�8� -  �QU  �?�E  �D"��  �C�TP�
  zF   Q(   @  �A  �?% � G-�d*<E��T=M(a&>U �;D"+:�`�` �u(`X`h�uxpx��  �?��*o�:�