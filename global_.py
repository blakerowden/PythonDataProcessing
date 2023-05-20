from enum import Enum


class FaultType(Enum):
    POLE3 = 3
    POLE2 = 2
    POLE1 = 1


PREFAULT = 1
FAULT = 2
POSTFAULT = 3


t = "t_s"
Ug_a = "Ug_a"
Ug_b = "Ug_b"
Ug_c = "Ug_c"
Ilsc_a = "Ilsc_a"
Ilsc_b = "Ilsc_b"
Ilsc_c = "Ilsc_c"
Imsc_a = "Imsc_a"
Imsc_b = "Imsc_b"
Imsc_c = "Imsc_c"
Is_a = "Is_a"
Is_b = "Is_b"
Is_c = "Is_c"
Ig_a = "Ig_a"
Ig_b = "Ig_b"
Ig_c = "Ig_c"
Udc = "Udc"
Ug_pos_d = "Ug_pos_d"
Ug_pos_q = "Ug_pos_q"
Ilsc_pos_d = "Ilsc_pos_d"
Ilsc_pos_q = "Ilsc_pos_q"
Imsc_pos_d = "Imsc_pos_d"
Imsc_pos_q = "Imsc_pos_q"
Is_pos_d = "Is_pos_d"
Is_pos_q = "Is_pos_q"
Ig_pos_d = "Ig_pos_d"
Ig_pos_q = "Ig_pos_q"
Ug_neg_d = "Ug_neg_d"
Ug_neg_q = "Ug_neg_q"
Ilsc_neg_d = "Ilsc_neg_d"
Ilsc_neg_q = "Ilsc_neg_q"
Imsc_neg_d = "Imsc_neg_d"
Imsc_neg_q = "Imsc_neg_q"
Is_neg_d = "Is_neg_d"
Is_neg_q = "Is_neg_q"
Ig_neg_d = "Ig_neg_d"
Ig_neg_q = "Ig_neg_q"
n = "n"
