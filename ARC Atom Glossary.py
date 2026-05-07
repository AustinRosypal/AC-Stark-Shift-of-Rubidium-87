from arc import *
atom = Rubidium()
# print(atom.getDipoleMatrixElement(5,0,0.5,-0.5,5,1,1.5,-1.5,-1))

# [5s_1/2, 4d_5/2, 5d_5/2, 6d_5/2, 7d_5/2, 8d_5/2]  for 5p_3/2 analysis

print(atom.getReducedMatrixElementJ(5,1,1.5,5,0,0.5))
print(atom.getReducedMatrixElementJ(5,0,0.5,5,1,1.5))
print(atom.getReducedMatrixElementJ(5,1,1.5,4,2,2.5))
print(atom.getReducedMatrixElementJ(5,1,1.5,5,2,2.5))
print(atom.getReducedMatrixElementJ(5,1,1.5,6,2,2.5))
print(atom.getReducedMatrixElementJ(5,1,1.5,7,2,2.5))
print(atom.getReducedMatrixElementJ(5,1,1.5,8,2,2.5))