from skimage import io
from skimage.color import rgb2gray
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# import and scale data
chess = io.imread('chess.png')
chess_g = rgb2gray(chess)

scaler = MinMaxScaler()
jelly = io.imread('jellyfish.jpg')
jelly_g = rgb2gray(jelly)
scaler.fit(jelly_g)
jelly_g = scaler.transform(jelly_g)

scaler1 = MinMaxScaler()
ny = io.imread('newyork.jpg')
ny_g = rgb2gray(ny)
scaler1.fit(ny_g)
ny_g = scaler1.transform(ny_g)

print(chess_g.shape, jelly_g.shape, ny_g.shape)

# show images
fig1, axes1 = plt.subplots()
axes1.imshow(jelly_g, cmap = plt.cm.gray)
axes1.axis('off')
plt.savefig('jelly_g')

fig2, axes2 = plt.subplots()
axes2.imshow(chess_g, cmap = plt.cm.gray)
axes2.axis('off')
plt.savefig('chess_g')

fig3, axes3 = plt.subplots()
axes3.imshow(ny_g, cmap = plt.cm.gray)
axes3.axis('off')
plt.savefig('ny_g')

def singular_val_study(image):
    u, s, vT = np.linalg.svd(image)
    val_vec = np.arange(s.shape[0])
    var = np.zeros_like(s)
    var = (s*s)/np.sum(s*s)
    return val_vec, s, var


def svd_compress(image, rank):
    u, s, vT = np.linalg.svd(image)
    s = s[:rank]
    u = u[:,:rank]
    vT = vT[:rank,:]
    print(u.shape)
    print(s.shape)
    print(vT.shape)
    compress = (u @ np.diag(s)) @ vT
    return rank, compress

# CHECKERS
range, s, var = singular_val_study(chess_g)
fig1 = plt.figure()
plt.semilogy(range,s)
plt.xlabel('Position of Singular Value')
plt.ylabel('Value of Singular Value')
plt.xlim(0,200)
plt.savefig('checker_sv')

fig2 = plt.figure()
plt.scatter(range,var)
plt.xlabel('Position of Singular Value')
plt.ylabel('Variance explained')
plt.yscale('log')
plt.xlim(0,200)
plt.ylim(1e-36,2*1e0)
plt.savefig('checker_var')
plt.show()

rank, n_checker = svd_compress(chess_g, 2)
fig = plt.figure()
plt.imshow(n_checker, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Rank = 2')
plt.savefig('checker_compress')
plt.show()

# JELLY FISH
range, s, var = singular_val_study(jelly_g)
fig1 = plt.figure()
plt.semilogy(range,s)
plt.xlabel('Position of Singular Value')
plt.ylabel('Value of Singular Value')
plt.savefig('jelly_sv')

fig2 = plt.figure()
plt.scatter(range,var)
plt.xlabel('Position of Singular Value')
plt.ylabel('Variance explained')
plt.yscale('log')
plt.ylim(1e-10,2*1e0)
plt.savefig('jelly_var')
plt.show()

rank, n_jelly = svd_compress(jelly_g, 30)
fig = plt.figure()
plt.imshow(n_jelly, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Rank = 30')
plt.savefig('jelly_compress_30')
plt.show()

#NYC
range, s, var = singular_val_study(ny_g)
fig1 = plt.figure()
plt.semilogy(range,s)
plt.xlabel('Position of Singular Value')
plt.ylabel('Value of Singular Value')
plt.savefig('ny_sv')

fig2 = plt.figure()
plt.scatter(range,var)
plt.xlabel('Position of Singular Value')
plt.ylabel('Variance explained')
plt.yscale('log')
plt.ylim(1e-8,2*1e0)
plt.savefig('ny_var')
plt.show()

rank, n_ny = svd_compress(ny_g, 200)
fig = plt.figure()
plt.imshow(n_ny, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Rank = 200')
plt.savefig('ny_compress_200')
plt.show()














#
