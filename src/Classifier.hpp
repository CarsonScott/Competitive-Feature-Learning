#ifndef CLASSIFIER_HPP_INCLUDED
#define CLASSIFIER_HPP_INCLUDED

#include "Error.hpp"

struct Category
{
    float label;
    float error;
    Category(int l=0, float e=0)
    {
        label = l;
        error = e;
    }
};

struct CategorySet
{
    Array labels;
    Array errors;

    void append(float label, float error)
    {
        labels.push_back(label);
        errors.push_back(error);
    }

    void append(Category category)
    {
        append(category.label, category.error);
    }

    int size()
    {
        return labels.size();
    }

    Category operator [](int i)
    {
        return Category(labels[i], errors[i]);
    }
};

class Classifier
{
    int input_size;
    Array inputs;
    Array thresholds;
    Matrix features;
    Matrix weights;
    Matrix errors;
    Matrix previous_errors;
    Matrix progress;
    Matrix sample_sizes;
    CategorySet active_categories;
    float learning_rate;
    float decay_rate;

public:
    Classifier(int in)
    {
        input_size = in;
        learning_rate = 0.001;
        decay_rate = 0.0005;
        for(int i = 0; i < input_size; i++)
        {
            inputs.push_back(0);
        }
    }

    void create()
    {
        Array feature;
        Array weight;
        Array error;
        Array previous_error;
        Array prog;
        for(int i = 0; i < input_size; i++)
        {
             feature.push_back(0);
             weight.push_back(.1);
             error.push_back(0);
             previous_error.push_back(0);
             prog.push_back(0);
        }
        features.push_back(feature);
        weights.push_back(weight);
        errors.push_back(error);
        previous_errors.push_back(previous_error);
        progress.push_back(prog);
        thresholds.push_back(.5);
    }

    void create(Array feature, Array weight)
    {
        create();
        features.back() = feature;
        weights.back() = weight;
    }

    void updateError(int c, int f)
    {
        previous_errors[c][f] = errors[c][f];
        errors[c][f] = err(inputs[f], features[c][f]);
    }

    void updateProgress(int c, int f)
    {
        progress[c][f] = err(previous_errors[c][f], errors[c][f]);
    }

    void updateWeight(int c, int f)
    {
        weights[c][f] += learning_rate * errors[c][f]* (1-weights[c][f]);
    }

    void updateFeature(int c, int f)
    {
        features[c][f] += learning_rate * progress[c][f] * (1-weights[c][f]);
    }

public:
    CategorySet classify(Array in)
    {
        inputs = in;
        CategorySet categories;
        for(int c = 0; c < features.size(); c++)
        {
            float threshold = thresholds[c];
            float similarity = 1-wmse(features[c], inputs, weights[c]);

            if(similarity > threshold)
            {
                categories.append(c, similarity);
                thresholds[c] += learning_rate;
                if(thresholds[c] > 1)
                {
                    thresholds[c] = 1;
                }
            }
            else
            {
                thresholds[c] -= decay_rate;
                if(thresholds[c] < 0)
                {
                    thresholds[c] = 0;
                }
            }
        }
        active_categories = categories;
        return active_categories;
    }

    void update()
    {
        for(int i = 0; i < active_categories.size(); i++)
        {
            int c = active_categories[i].label;
            for(int f = 0; f < input_size; f++)
            {
                updateError(c, f);
                updateProgress(c, f);
                updateWeight(c, f);
                updateFeature(c, f);

                if(weights[c][f] < 0)
                {
                    weights[c][f] = 0;
                }
                else if(weights[c][f] > 1)
                {
                    weights[c][f] = 1;
                }
            }
        }
    }
};

#endif // CLASSIFIER_HPP_INCLUDED
