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

    void insert(int i, float label, float error)
    {
        labels.insert(labels.begin() + i, label);
        errors.insert(errors.begin() + i, error);

    }

    void erase(int i)
    {
        labels.erase(labels.begin() + i);
        errors.erase(errors.begin() + i);
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

    int get_sorted_index(Array arr, float val)
    {
        int index = 0;
        while(index < arr.size())
        {
            if(val <= arr[index])
            {
                return index;
            }
            else
            {
                index += 1;
            }
        }
        return arr.size();
    }

public:
    Classifier(){}
    Classifier(int in_size, int cls_size)
    {
        input_size = in_size;
        class_size = cls_size;
        learning_rate = 0.0001;
        decay_rate = 0.00001;
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

    void updateWeight(int c, int f)
    {
        weights[c][f] += learning_rate * ((progress[c][f]+1) / (1+errors[c][f]));
    }

    void updateFeature(int c, int f)
    {
        features[c][f] += learning_rate * errors[c][f];
    }

public:
    CategorySet classify(Array in)
    {
        inputs = in;
        CategorySet categories;

        for(int c = 0; c < features.size(); c++)
        {
            float similarity = 1-wmse(features[c], inputs, weights[c]);
            float threshold = thresholds[c];

            if(similarity >= threshold)
            {
                float error = 1 - similarity;
                int index = get_sorted_index(categories.errors, error);
                if(categories.errors.size() > 0)
                {
                    if(index < categories.errors.size())
                    {
                        categories.insert(index, c, error);
                        if(categories.errors.size() > class_size)
                        {
                            categories.erase(categories.errors.size()-1);
                        }
                    }
                }
                else
                {
                    categories.append(c, error);
                }
            }
            thresholds[c] -= decay_rate;
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
            thresholds[c] += learning_rate;
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


    Array decode(CategorySet c)
    {
        Array rep;
        for(int i = 0; i < inputs.size(); i++)
        {
            float val = 0;

            if(c.labels.size() > 0)
            {
                for(int j = 0; j < c.labels.size(); j++)
                {
                    int index = c.labels[j];
                    val += features[index][i] * weights[index][i];
                }
                val /= c.labels.size();
            }

            rep.push_back(val);
        }

        return rep;
    }
};

#endif // CLASSIFIER_HPP_INCLUDED
