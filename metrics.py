def f1_s(P, R):
    f1_s = (2 * P * R) / (P + R)
    return f1_s

def ious(tp, fp, fn):
    IoU = tp / (tp + fp + fn)
    return IoU

ap, ar, atp, afp, afn, c1p, c2p, c3p, c4p, c5p, c6p, c1r, c2r, c3r, c4r, c5r, c6r = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0





c1 = f1_s(c1p, c1r)
c2 = f1_s(c2p, c2r)
c3 = f1_s(c3p, c3r)
c4 = f1_s(c4p, c4r)
c5 = f1_s(c5p, c5r)
c6 = f1_s(c6p, c6r)
af = f1_s(ap, ar)
aiou = ious(atp, afp, afn)




print(af, aiou)
print(c1, c2, c3, c4, c5, c6)
