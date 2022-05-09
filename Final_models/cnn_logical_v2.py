import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from basic_model import BasicModel

class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 16, 16*4*4)
    
# class mlp_module(nn.Module):
#     def __init__(self):
#         super(mlp_module, self).__init__()
#         self.fc1 = nn.Linear(32*4*4, 512)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(512, 8)
#         self.dropout = nn.Dropout(0.5)
        
#     def forward(self, x):
#         x = self.relu1(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x



class CNN_Logical(BasicModel):
    def __init__(self, args):
        super(CNN_Logical, self).__init__(args)
        self.conv = conv_module()
        self.emb_dim = 16*4*4
        self.true_vector = torch.nn.Parameter(torch.from_numpy(
        np.random.uniform(0, 0.1, size=self.emb_dim).astype(np.float32)),
        requires_grad=False) 
        self.softmax = nn.Softmax(dim=1)

        # Layers of NOT network
        self.not_layer_1 = torch.nn.Linear(self.emb_dim, self.emb_dim)
        self.not_layer_2 = torch.nn.Linear(self.emb_dim, self.emb_dim)

        # Layers of AND network
        self.and_layer_1 = torch.nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.and_layer_2 = torch.nn.Linear(self.emb_dim, self.emb_dim)
        
        self.dropout_layer_logic = torch.nn.Dropout(0.1)
        
        
        # Layers of create_row network
        self.create_row_layer_1 = torch.nn.Linear(3 * self.emb_dim, self.emb_dim)
        self.create_row_layer_2 = torch.nn.Linear(self.emb_dim, self.emb_dim)
        self.create_row_layer_dropout = torch.nn.Dropout(0.1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
    
    
    def logic_not(self, vector):
        vector = F.relu(self.not_layer_1(vector))
        if self.training:
            vector = self.dropout_layer_logic(vector)
        out = self.not_layer_2(vector)
        return out

    def logic_and(self, vector1, vector2, dim=1):
        vector = torch.cat((vector1, vector2), dim)
        vector = F.relu(self.and_layer_1(vector))
        if self.training:
            vector = self.dropout_layer_logic(vector)
        out = self.and_layer_2(vector)
        return out
    
    
    def logic_or(self, vector1, vector2, dim=1):
        vector = torch.cat((vector1, vector2), dim)
        vector = F.relu(self.and_layer_1(vector))
        if self.training:
            vector = self.dropout_layer_logic(vector)
        out = self.and_layer_2(vector)
        return out
    
    def create_row(self, vector1, vector2,vector3, dim=1):
        vector = torch.cat((vector1, vector2,vector3), dim)
        vector = F.relu(self.create_row_layer_1(vector))
        if self.training:
            vector = self.create_row_layer_dropout(vector)
        out = self.create_row_layer_2(vector)
        return out
    
    
    def mse(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2, train=False)
        return (vector1 - vector2) ** 2

    def dot_product(self, vector1, vector2):
        vector1, vector2 = self.uniform_size(vector1, vector2, train=False)
        result = (vector1 * vector2).sum(dim=-1)
        vector1_pow = vector1.pow(2).sum(dim=-1).pow(self.sim_alpha)
        vector2_pow = vector2.pow(2).sum(dim=-1).pow(self.sim_alpha)
        result = result / torch.clamp(vector1_pow * vector2_pow, min=1e-8)
        return result

    def similarity(self, vector1, vector2, sigmoid=False):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * 100
        if sigmoid:
            return result.sigmoid()
        return result    

    
    def compute_loss(self, output, target, meta_target):
        pred = output[0]
        constraints = output[1]
        true_constraints = output[2]
        
        false_vector = self.logic_not(self.true_vector)  # we compute the representation
        # for the FALSE vector

        # here, we need to implement the logical regularizers for the logical regularization

        # here, we implement the logical regularizers for the NOT operator

        # minimizing 1 - similarity means maximizing the cosine similarity between the two event vectors in input
        # minimizing 1 + similarity means minimizing the cosine similarity between the two event vectors in input

        # here, we maximize the similarity between not not true and true
        r_not_not_true = (1 - F.cosine_similarity(
            self.logic_not(self.logic_not(self.true_vector)), self.true_vector,dim=0))

        # here, we maximize the similarity between not true and false
        #r_not_true = (1 - F.cosine_similarity(self.logic_not(self.true_vector), false_vector, dim=0))

        # here, we maximize the similarity between not not x and x
        r_not_not_self = \
            (1 - F.cosine_similarity(self.logic_not(self.logic_not(constraints)), constraints)).mean()

        # here, we minimize the similarity between not x and x
        r_not_self = (1 + F.cosine_similarity(self.logic_not(constraints), constraints)).mean()

        # here, we minimize the similarity between not not x and not x
        r_not_not_not = \
            (1 + F.cosine_similarity(self.logic_not(self.logic_not(constraints)),
                                     self.logic_not(constraints))).mean()

        # here, we implement the logical regularizers for the OR operator

        # here, we maximize the similarity between x OR True and True
        r_or_true = (1 - F.cosine_similarity(
            self.logic_or(constraints, self.true_vector.expand_as(constraints)),
            self.true_vector.expand_as(constraints))).mean()

        # here, we maximize the similarity between x OR False and x
        r_or_false = (1 - F.cosine_similarity(
            self.logic_or(constraints, false_vector.expand_as(constraints)), constraints)).mean()

        # here, we maximize the similarity between x OR x and x
        r_or_self = (1 - F.cosine_similarity(self.logic_or(constraints, constraints), constraints)).mean()

        # here, we maximize the similarity between x OR not x and True
        r_or_not_self = (1 - F.cosine_similarity(
            self.logic_or(constraints, self.logic_not(constraints)),
            self.true_vector.expand_as(constraints))).mean()

        # same rule as before, but we flipped operands
        r_or_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_or(self.logic_not(constraints), constraints),
            self.true_vector.expand_as(constraints))).mean()

        # here, we implement the logical regularizers for the AND operator

        # here, we maximize the similarity between x AND True and x
        r_and_true = (1 - F.cosine_similarity(
            self.logic_and(constraints, self.true_vector.expand_as(constraints)), constraints)).mean()

        # here, we maximize the similarity between x AND False and False
        r_and_false = (1 - F.cosine_similarity(
            self.logic_and(constraints, false_vector.expand_as(constraints)),
            false_vector.expand_as(constraints))).mean()

        # here, we maximize the similarity between x AND x and x
        r_and_self = (1 - F.cosine_similarity(self.logic_and(constraints, constraints), constraints)).mean()

        # here, we maximize the similarity between x AND not x and False
        r_and_not_self = (1 - F.cosine_similarity(
            self.logic_and(constraints, self.logic_not(constraints)),
            false_vector.expand_as(constraints))).mean()

        # same rule as before, but we flipped operands
        r_and_not_self_inverse = (1 - F.cosine_similarity(
            self.logic_and(self.logic_not(constraints), constraints),
            false_vector.expand_as(constraints))).mean()

        # True/False rule

        # here, we minimize the similarity between True and False
        true_false = 1 + F.cosine_similarity(self.true_vector, false_vector, dim=0)

        r_loss = r_not_not_true + r_not_not_self + r_not_self + r_not_not_not + \
                 r_or_true + r_or_false + r_or_self + r_or_not_self + r_or_not_self_inverse + true_false + \
                 r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse
        
        rows_and_true= (1 - F.cosine_similarity(
            self.true_vector.expand_as(true_constraints), true_constraints)).mean()
        
        
        
        
        loss = F.cross_entropy(pred, target) + r_loss*0.1 + rows_and_true*0.1
        return loss

        
    
    
    def forward(self, x):
        alpha = 1.0
        constraints = []
        true_constraints = []
        image_features = self.conv(x.view(-1, 1, 80, 80))
#         print(image_features.shape)
        
    
    
        row1 = self.create_row(image_features[:,0],image_features[:,1],image_features[:,2])
        constraints.append(row1)
        true_constraints.append(row1)
        
        row2 = self.create_row(image_features[:,3],image_features[:,4],image_features[:,5])
        constraints.append(row2)
        true_constraints.append(row2)
        
        
        
        row1_row2 = self.logic_and(row1,row2)
        constraints.append(row1_row2)
        true_constraints.append(row1_row2)
        
        n_row1_row2_or = self.logic_or(self.logic_not(row1),self.logic_not(row2))
        constraints.append(n_row1_row2_or)
#         true_constraints.append(row1_row2_or)
        
        
        row3_1 = self.create_row(image_features[:,6],image_features[:,7],image_features[:,8])
        row3_1_n = self.logic_not(row3_1)
        choice1 = self.logic_or(n_row1_row2_or,row3_1)
        constraints.append(row3_1)
        constraints.append(row3_1_n)
        constraints.append(choice1)
        
        row3_2 = self.create_row(image_features[:,6],image_features[:,7],image_features[:,9])
        row3_2_n = self.logic_not(row3_2)
        choice2 = self.logic_or(n_row1_row2_or,row3_2)
        constraints.append(row3_2)
        constraints.append(row3_2_n)
        constraints.append(choice2)
        
        row3_3 = self.create_row(image_features[:,6],image_features[:,7],image_features[:,10])
        row3_3_n = self.logic_not(row3_3)
        choice3 = self.logic_or(n_row1_row2_or,row3_3)
        constraints.append(row3_3)
        constraints.append(row3_3_n)
        constraints.append(choice3)
        
        row3_4 = self.create_row(image_features[:,6],image_features[:,7],image_features[:,11])
        row3_4_n = self.logic_not(row3_4)
        choice4 = self.logic_or(n_row1_row2_or,row3_4)
        constraints.append(row3_4)
        constraints.append(row3_4_n)
        constraints.append(choice4)
        
        row3_5 = self.create_row(image_features[:,6],image_features[:,7],image_features[:,12])
        row3_5_n = self.logic_not(row3_5)
        choice5 = self.logic_or(n_row1_row2_or,row3_5)
        constraints.append(row3_5)
        constraints.append(row3_5_n)
        constraints.append(choice5)
        
        row3_6 = self.create_row(image_features[:,6],image_features[:,7],image_features[:,13])
        row3_6_n = self.logic_not(row3_6)
        choice6 = self.logic_or(n_row1_row2_or,row3_6)
        constraints.append(row3_6)
        constraints.append(row3_6_n)
        constraints.append(choice6)
        
        row3_7 = self.create_row(image_features[:,6],image_features[:,7],image_features[:,14])
        row3_7_n = self.logic_not(row3_7)
        choice7 = self.logic_or(n_row1_row2_or,row3_7)
        constraints.append(row3_7)
        constraints.append(row3_7_n)
        constraints.append(choice7)
        
        row3_8 = self.create_row(image_features[:,6],image_features[:,7],image_features[:,15])
        row3_8_n = self.logic_not(row3_8)
        choice8 = self.logic_or(n_row1_row2_or,row3_8)
        constraints.append(row3_8)
        constraints.append(row3_8_n)
        constraints.append(choice8)

        
        
        
#         constraints.append(image_features.view(-1,256))
        if len(constraints) > 0:
#             for c in constraints:
#                 print(c.shape)
            constraints = torch.cat(constraints, dim=0)
#             print(constraints.shape)
            constraints = constraints.view(-1, self.emb_dim)
            true_constraints = torch.cat(true_constraints, dim=0)
#             print(constraints.shape)
            true_constraints = true_constraints.view(-1, self.emb_dim)
        prediction_c1 = self.similarity(choice1, self.true_vector).view([-1])
        prediction_c2 = self.similarity(choice2, self.true_vector).view([-1])
        prediction_c3 = self.similarity(choice3, self.true_vector).view([-1])
        prediction_c4 = self.similarity(choice4, self.true_vector).view([-1])
        prediction_c5 = self.similarity(choice5, self.true_vector).view([-1])
        prediction_c6 = self.similarity(choice6, self.true_vector).view([-1])
        prediction_c7 = self.similarity(choice7, self.true_vector).view([-1])
        prediction_c8 = self.similarity(choice8, self.true_vector).view([-1])
#         print(constraints.shape)
#         print(prediction_c1.shape)
        
        score = torch.stack((prediction_c1,prediction_c2,prediction_c3,prediction_c4,prediction_c5,prediction_c6,
                             prediction_c7,prediction_c8),-1) 
#         print(score.shape)
#         score = score.view(-1,8)
#         score = self.softmax(score)
        
        return score, constraints, true_constraints

    
    ## Check similarity with row1_row2 , choice is just row3 -> similarity row1_row2, row3, softmax
    ## row1_row2 = 1; row1_row3(correct_choice) = 1 , row2_row3(correct_choice) = 1; 14 wrong s  
    ## Concat images and add mlp; 3d -> d
    
   # Formulation - 2
#     I1, I2 -> I3
#    R1, I4, I5 -> I6
#    R1, R2, I7, I8 -> ?
#  ~R1 v ~R2 v ~R3 = T

