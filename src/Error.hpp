#ifndef ERROR_HPP_INCLUDED
#define ERROR_HPP_INCLUDED

#include <vector>
#include <cmath>

#define Array std::vector<float>
#define Matrix std::vector<Array>

float err(float x1, float x2)
{
    float e = x1-x2;
    return e;
}

float werr(float x1, float x2, float w)
{
    float e = err(x1, x2) * w;
    return e;
}

float mse(float x1, float x2)
{
    float e = pow(err(x1, x2), 2);
    return e;
}

float wmse(float x1, float x2, float w)
{
    float e = mse(x1, x2) * w;
    return e;
}

float err(Array x1, Array x2)
{
    float e = 0;
    for(int i = 0; i < x1.size(); i++)
    {
        e = e+err(x1[i], x2[i]);
    }
    e = e / x1.size();
    return e;
}

float werr(Array x1, Array x2, Array w)
{
    float e = 0;
    for(int i = 0; i < x1.size(); i++)
    {
        e = e+werr(x1[i], x2[i], w[i]);
    }
    e = e / x1.size();
    return e;
}

float mse(Array x1, Array x2)
{
    float e = 0;
    for(int i = 0; i < x1.size(); i++)
    {
        e = e+mse(x1[i], x2[i]);
    }
    e = e / x1.size();
    return e;
}

float wmse(Array x1, Array x2, Array w)
{
    float e = 0;
    for(int i = 0; i < x1.size(); i++)
    {
        e = e+wmse(x1[i], x2[i], w[i]);
    }
    e = e / x1.size();
    return e;
}

#endif // ERROR_HPP_INCLUDED
