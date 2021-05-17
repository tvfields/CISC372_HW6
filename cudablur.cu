#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define BSIZE = 256;

__global__ void computeRow(float *src, float *dest, int pWidth, int radius, int bpp, int height){
    int i; 

    int bradius = radius * bpp;
     
    int row = threadIdx.x + (blockIdx.x * blockDim.x);

    if (height > row){
        for (i = 0; i < bpp; i++){
            dest[row * pWidth + i] = src[row * pWidth + i];
        }

        for (i = bpp; i < bradius * 2 * bpp; i++){
            dest[row * pWidth + i] = src[row * pWidth + i] + dest[row * pWidth + i - bpp];
        }

        for (i = bradius * 2 + bpp; i < pWidth; i++){
            dest[row * pWidth + i] = src[row * pWidth + i] + dest[row * pWidth + i - bpp] - src[row * pWidth + i - 2 * bradius - bpp];
        }

        for (i = bradius; i < pWidth; i++){
            dest[row * pWidth + i - bradius] = dest[row * pWidth + i] / (radius * 2 + 1);
        }
   
        for (i = 0; i < bradius; i++){
            dest[row * pWidth + i] = 0;
            dest[(row + 1) * pWidth - 1 - i] = 0;
        }
    }
}

__global__ void computeColumn(uint8_t *src, float *dest, int pWidth, int height, int radius, int bpp){
    int i;
    int col = threadIdx.x + (blockIdx.x * blockDim.x);

    if (pWidth > col){
        dest[col] = src[col];

        for (i = 1; i <= radius * 2; i++){
            dest[i * pWidth + col] = src[i * pWidth + col] + dest[(i - 1) * pWidth + col];
        }
        
        for (i = radius * 2 + 1; i < height; i++){
            dest[i * pWidth + col] = src[i * pWidth + col] + dest[(i - 1) * pWidth + col] - src[(i - 2 * radius - 1) * pWidth + col];
        }

        for (i = radius; i < height; i++){
            dest[(i - radius) * pWidth + col] = dest[i * pWidth + col] / (radius * 2 + 1);
        }

        for (i = 0; i < radius; i++)
        {
            dest[i * pWidth + col] = 0;
            dest[(height - 1) * pWidth - i * pWidth + col] = 0;
        }
    }
}

int Usage(char *name)
{
    printf("%s: <filename> <blur radius>\n\tblur radius=pixels to average on any side of the current pixel\n", name);
    return -1;
}

int main(int argc, char **argv){
    long t1, t2;
    int r = 0;
    int i;
    int w, h, bpp, pw;
    char *fname;
    uint8_t *img;
    float *dest, *mid;
    uint8_t *dest_img;

    if (argc != 3){
        return Usage(argv[0]);
    }

    fname = argv[1];
    sscanf(argv[2], "%d", &r);
    img = stbi_load(filename, &w, &h, &bpp, 0);

    pw = w * bpp;

    cudaMalloc(&dest_img, sizeof(uint8_t) * pw * h);
    cudaMallocManaged(&mid, sizeof(float) * pw * h);
    cudaMallocManaged(&dest, sizeof(float) * pw * h);

    int bnum = (pw + (BSIZE - 1)) / BSIZE;

    t1 = clock();
    computeColumn<<<bnum, BSIZE>>>(dest_img, mid, pw, h, r, bpp);
    cudaDeviceSynchronize();
    stbi_image_free(img); 

    bnum = (h + (BS - 1)) / BSIZE;

    computeRow<<<bnum, BSIZE>>>(mid, dest, pw, r, bpp, h);
    cudaDeviceSynchronize();
    t2 = clock();
    cudaFree(mid); 
    img = (uint8_t*)malloc(sizeof(uint8_t) * pw * h);
    
    for (i = 0; i < pw * h; i++){
        img[i] = (uint8_t)dest[i];
    }
    
    stbi_write_png("output.png", w, h, bpp, img, bpp * w);
    cudaFree(img);
    cudaFree(mid);
    cudaFree(dest);
    free(img);
    printf("Blur with radius %d complete in %f seconds\n", r, (double)(t2 - t1) / (double)CLOCKS_PER_SEC);
}
