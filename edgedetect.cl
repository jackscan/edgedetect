
constant float4 filter = { 0.09375f, 0.3125f, 0.09375f, 0.f };

#if __IMAGE_SUPPORT__

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
//const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

constant float4 gray = { 0.33f, 0.33f, 0.33f, 0.0f };

float3 readRow(read_only image2d_t src, int x, int y)
{
   return (float3)(
      dot(read_imagef(src, sampler, (int2)(x-1, y)), gray),
      dot(read_imagef(src, sampler, (int2)(x, y)), gray),
      dot(read_imagef(src, sampler, (int2)(x+1, y)), gray));
}

float3 readCol(read_only image2d_t src, int x, int y)
{
   return (float3)(
      dot(read_imagef(src, sampler, (int2)(x, y-1)), gray),
      dot(read_imagef(src, sampler, (int2)(x, y)), gray),
      dot(read_imagef(src, sampler, (int2)(x, y+1)), gray));
}

kernel void edgeDetect(read_only image2d_t src, write_only image2d_t dst)
{
   int x = get_global_id(0);
   int y = get_global_id(1);

   float2 g = {
      dot(readCol(src, x-1, y), filter.xyz) - dot(readCol(src, x+1, y), filter.xyz),
      dot(readRow(src, x, y-1), filter.xyz) - dot(readRow(src, x, y+1), filter.xyz)};
   float a = atan2pi(g.x, g.y) * 3.f / 2.f + 3.f / 2.f;
   float l = length(g);

   float i;
   float b = fract(a, &i);
   int i0 = ((int)i) % 3;
   int i1 = (i0 + 1) % 3;
   float c[3] = { 0, 0, 0 };
   //printf("(%f, %i, %i, %f)", a, i0, i1, b);
   c[i0] = 1-b;
   c[i1] = b;
   float3 rgb = normalize((float3)(c[0], c[1], c[2])) * l;

   write_imagef(dst, (int2)(x, y), (float4)(rgb, 1));
}

#endif

float convertToGray(uchar4 rgba)
{
   const float4 gray = { 0.00130719f, 0.00130719f, 0.00130719f, 0.f };
   return dot(convert_float4(rgba), gray);
}

float4 readUcharRow(global uchar4 *src, int x, int y, int width)
{
   return (float4)(
      convertToGray(src[x-1 + y*width]),
      convertToGray(src[x   + y*width]),
      convertToGray(src[x+1 + y*width]),
      0.f);
}

float4 readUcharCol(global uchar4 *src, int x, int y, int width)
{
   return (float4)(
      convertToGray(src[x-1 + y*width]),
      convertToGray(src[x   + y*width]),
      convertToGray(src[x+1 + y*width]),
      0.f);
}

uchar4 convertToUchar4(float4 rgba)
{
   const float4 f = { 255.9999f, 255.9999f, 255.9999f, 255.f };
   return convert_uchar4_rtz(rgba * f);
}

kernel void bufferEdgeDetect(global uchar4 *src, global uchar4 *dst, int width)
{
   int x = get_global_id(0);
   int y = get_global_id(1);

   float2 g = {
      dot(readUcharCol(src, x-1, y, width), filter) - dot(readUcharCol(src, x+1, y, width), filter),
      dot(readUcharRow(src, x, y-1, width), filter) - dot(readUcharRow(src, x, y+1, width), filter)};
   float a = atan2pi(g.x, g.y) * 3.f / 2.f + 3.f / 2.f;
   float l = length(g);

   float i;
   float b = fract(a, &i);
   int i0 = ((int)i) % 3;
   int i1 = (i0 + 1) % 3;
   float c[3] = { 0, 0, 0 };
   //printf("(%f, %i, %i, %f)", a, i0, i1, b);
   c[i0] = 1-b;
   c[i1] = b;
   float4 rgba = normalize((float4)(c[0], c[1], c[2], 0.f)) * l;
   rgba.w = 1.f;

   dst[x + y*width] = convertToUchar4(rgba);
}
