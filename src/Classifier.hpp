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
    int class_size;
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
    Classifier(){}
    Classifier(int in_size, int cls_size)
    {
        input_size = in_size;
        class_size = cls_size;
        learning_rate = 0.0001;
        decay_rate = 0.0001;
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
             weight.push_back(.5);
             error.push_back(0);
             previous_error.push_back(1);
             prog.push_back(0);
        }
        features.push_back(feature);
        weights.push_back(weight);
        errors.push_back(error);
        previous_errors.push_back(previous_error);
        progress.push_back(prog);
        thresholds.push_back(.5);
    }

    void create(Array feature, Array weight, float threshold)
    {
        create();
        features.back() = feature;
        weights.back() = weight;
        thresholds.back() = threshold;
    }

    void updateError(int c, int f)
    {
        previous_errors[c][f] = errors[c][f];
        errors[c][f] = werr(inputs[f], features[c][f], weights[c][f]);
    }

    void updateProgress(int c, int f)
    {
        progress[c][f] = werr(previous_errors[c][f], errors[c][f], weights[c][f]);
    }

    void updateWeight(int c, int f, float error)
    {
        weights[c][f] += learning_rate * progress[c][f]*(error);
    }

    void updateFeature(int c, int f, float error)
    {
        features[c][f] += learning_rate * errors[c][f]*(error);
    }

public:
    CategorySet classify(Array in)
    {
        inputs = in;
        CategorySet categories;

        int classes = 0;
        int worst_feature = 0;
        for(int c = 0; c < features.size(); c++)
        {
            float similarity = 1-wmse(features[c], inputs, weights[c]);
            float threshold = thresholds[c];

            if(similarity > threshold || categories.labels.size() == 0)
            {
                if(classes < class_size)
                {
                    categories.append(c, 1-similarity);
                    if(1-similarity > categories.errors[worst_feature])
                    {
                        worst_feature = categories.errors.size()-1;
                    }
                    classes ++;
                }
                else
                {
                    categories.labels[worst_feature] = c;
                    categories.errors[worst_feature] = 1-similarity;

                    for(int f = 0; f < categories.errors.size(); f++)
                    {
                        if(categories.errors[f] > categories.errors[worst_feature])
                            worst_feature = f;
                    }
                }
            }
            thresholds[c] += decay_rate*(err(similarity, threshold));
            if(thresholds[c] > 1)
            {
                thresholds[c] = 1;
            }
            else if(thresholds[c] < 0)
            {
                thresholds[c] = 0;
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
                updateWeight(c, f, active_categories[i].error);
                updateFeature(c, f, active_categories[i].error);

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
