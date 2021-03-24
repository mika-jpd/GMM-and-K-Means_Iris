import numpy as np
import pandas as pd
import math
import csv
import KMeans as km


class GMM:
    def __init__(self, str_dataset, k):
        self.data = self.load_data(str_dataset)
        #clustering numbers
        self.k = np.empty(k , dtype=int)
        for i in range(0, k):
            self.k[i] = i

    def GMM(self):

        #innitialization
        log_likes = np.zeros(len(self.k))

        mix_coef = np.zeros(len(self.k))
        mean = np.zeros((len(self.k),len(self.data[0])))
        labels = np.zeros(len(self.data))

        cov = []
        for i in range(0, len(self.k)):
            cov.append(np.zeros((len(self.data[0]), len(self.data[0]))))

        #init the mix coeficients
        prob_start = 1/len(self.k)
        mix_coef = np.array(mix_coef)

        #init the clusters & labels
        cluster_start = np.array_split(self.data, len(self.k))


        for i in range(0, len(self.k)):
            mean[i] = self.init_mean(cluster_start[i])
            cov[i] = np.cov(cluster_start[i].T)
            mix_coef[i] = prob_start

        x = True

        # calculate log likelihood for comparison after the iteration
        log_l_prev = self.log_likelihood(self.data, mean, cov, mix_coef)
        #while loop until convergence
        while(x):
            #assign cluster to each point
            for i in range(0, len(self.data)):
                labels[i] = self.assign_cluster(self.data[i], mean, cov, mix_coef)

            self.optimize_mean(self.data, mean, cov, mix_coef)
            self.optimize_cov(self.data, mean, cov, mix_coef)
            self.optimize_mix_coef(self.data, mix_coef, mean, cov)

            log_l_post = self.log_likelihood(self.data, mean, cov, mix_coef)
            if(math.fabs((log_l_post-log_l_prev))<(10**(-5))):
                x = False
                return labels
            else:
                log_l_prev = log_l_post

    #finds the mean of a cluster given an entire set of datapoints
    def optimize_mean(self, data, mean, cov, mix_coef):
        #iterate through all the clusters to find each clusters mean
        for k in range(0, len(mean)):
            numerator = 0
            denominator = 0
            #iterate through all the points
            for i in range(0, len(data)):
                numerator += (self.r_score(data[i],mean,cov,mix_coef,k))*data[i]
                denominator += (self.r_score(data[i],mean,cov,mix_coef,k))
            mean[k] = numerator/denominator

    #finds the cov of a cluster
    def optimize_cov(self, data, mean, cov, mix_coef):
        #iterate through all the clusters
        for k in range(0, len(cov)):
            numerator = 0
            denominator = 0
            # iterate through all the points
            for i in range(0, len(data)):
                #numerator += self.r_score(data[i], mean, cov, mix_coef, k)*
                numerator += self.r_score(data[i], mean, cov, mix_coef, k) * (np.subtract(data[i], mean[k]))*np.subtract(data[i], mean[k])[np.newaxis,:].T
                denominator += self.r_score(data[i], mean, cov, mix_coef, k)
            cov[k] = numerator/denominator

    #finds the mix_coef of each custer
    def optimize_mix_coef(self, data, mix_coef, mean, cov):
        denominator = len(data)
        for k in range(0, len(mix_coef)):
            numerator = 0
            for i in range(0, len(data)):
                numerator += self.r_score(data[i],mean, cov, mix_coef,k)
            mix_coef[k] = numerator/denominator

    #assigns the point to a cluster
    def assign_cluster(self, point, mean, cov, mix_coef):
        closest = float('-inf')
        decision = 0
        for i in range(0, len(mean)):
            x = self.r_score(point, mean, cov, mix_coef, i)
            if((x)>closest):
                closest = x
                decision = i
        return decision

    #takes the r score of a point where index is the k of r(x,k)
    def r_score(self, point, mean, cov, mix_coef, index):
        nominator = self.weighted_normal_function(point, mean[index], cov[index], mix_coef[index])
        denominator = 0
        for j in range(0, len(mean)):
            denominator += self.weighted_normal_function(point, mean[j], cov[j], mix_coef[j])
        return nominator/denominator

    def log_likelihood(self, data,mean, cov, mix_coef):
        log_l = 0
        log_temp = 1 #for numerical stability
        for i in range(0, len(data)):
            log_l += np.log(log_temp)
            log_temp = 0
            for k in range(0, len(mix_coef)):
                #lab = label[i]
                log_temp += (self.weighted_normal_function(data[i], mean[k], cov[k], mix_coef[k]))
        return log_l

    #Calculates the weighted norm
    #point is n*1, mean is n*1, cov is n*n, mix_coef is a number
    def weighted_normal_function(self, point, mean, cov, mix_coef):
        #return mix_coef*(2*np.pi)**(-len(cov)/2)*np.linalg.det(cov)**(-1/2)*np.exp((-1/2)*(np.subtract(point, mean)).T*np.linalg.inv(cov)*(np.subtract(point, mean)))
        return mix_coef*(2*np.pi)**(-len(cov)/2)*np.linalg.det(cov)**(-1/2)*np.exp((-1/2)*np.dot((np.subtract(point, mean).T), np.dot(np.linalg.inv(cov),(np.subtract(point, mean)))))


    def init_mean(self, data):
        data = np.array(data)
        numerater = np.zeros(data[0].shape)
        denominator = len(data)

        for i in data:
            numerater = np.add(numerater, i)
        return numerater/denominator

    def load_data(self, path):
        dataset = np.genfromtxt(path, delimiter='\t')
        return dataset


x = GMM('Data.tsv', 3)
labels = x.GMM()
with open('GMM_output.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(labels)
