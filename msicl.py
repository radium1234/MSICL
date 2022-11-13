import os.path
import numpy as np
import json
import array
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import faiss
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from sentence_transformers import SentenceTransformer



class MSICL(GeneralRecommender):
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset, args):
        super(MSICL, self).__init__(config, dataset)
        # load dataset info
        self.dataset=config["dataset"]
        self.imagek=args.imagek
        self.textk=args.textk
        self.image=args.image
        self.text=args.text
        self.filterhop=args.hop
        self.filtercluster=args.cluster

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        if self.image==1:
            if not os.path.exists('dataset/'+self.dataset+'/image_feat.npy'):
                self.image_feat()
            self.imagefeat = np.load('dataset/'+self.dataset+'/image_feat.npy').astype('float32')
            l=self.imagefeat.shape[0]
            if self.n_items!=l+1:
                self.image_feat()
                self.imagefeat = np.load('dataset/' + self.dataset + '/image_feat.npy').astype('float32')
                l = self.imagefeat.shape[0]
            assert self.n_items==l+1


        if self.text==1:
            if not os.path.exists('dataset/'+self.dataset+'/text_feat.npy'):
                self.text_feat()
            self.textfeat = np.load('dataset/'+self.dataset+'/text_feat.npy').astype('float32')
            l = self.textfeat.shape[0]
            if self.n_items != l+ 1:
                self.text_feat()
                self.textfeat = np.load('dataset/' + self.dataset + '/text_feat.npy').astype('float32')
                l = self.textfeat.shape[0]
            assert self.n_items == l+1



        # load parameters info
        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']          # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']      # float32 type: the weight decay for l2 normalization

        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']
        self.ssl_weight=config["ssl_weight"]
        self.text_weight=args.text_weight
        self.image_weight = args.image_weight

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim,padding_idx=0)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim,padding_idx=0)


        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        self.norm_adj_mat = self.get_norm_adj_mat().to(self.device)
        if self.filterhop>0:
            self.make_adjdic()
        if self.filtercluster > 0:
            self.make_cluster()
        if self.image==1:
            self.imagelist = self.knn(self.imagefeat, 4096, self.imagek)
        if self.text == 1:
            self.textlist = self.knn(self.textfeat, 768, self.textk)
        torch.cuda.empty_cache()
        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']
    def make_adjdic(self):
        a=self.norm_adj_mat.to_dense()
        self.sum=0
        b= a
        for i in range(0,self.filterhop*2-1):
            b=torch.mm(b,a)
            if i%2==0:
                self.sum=self.sum+b
                print(self.sum)
        self.sum=self.sum.cpu()
    def make_cluster(self):
        if os.path.exists('dataset/'+self.dataset+f'/cluster{self.filtercluster}.npy'):
            self.cluster=np.load('dataset/'+self.dataset+f'/cluster{self.filtercluster}.npy')
        else:
            x=np.load('dataset/'+self.dataset+'/lightrecord.npy')
            kmeans = faiss.Kmeans(d=64, k=self.filtercluster, gpu=True)
            kmeans.train(x)
            _, I = kmeans.index.search(x, 1)
            np.save('dataset/'+self.dataset+f'/cluster{self.filtercluster}.npy',I)
            self.cluster=I



    def knn(self,modalfeat,d,k):
        index = faiss.IndexFlatL2(d)
        faiss.normalize_L2(modalfeat)
        index.add(modalfeat)

        _, I = index.search(modalfeat, k)
        I=(I + 1)[:, 1:]
        if self.filterhop==0 and self.filtercluster==0:
            return I
        count1=0
        if self.filterhop==0:
            for i in range(1,self.n_items):
                if(i%100==0):
                    print (i)
                    print(f"count1avg={count1 / i}")
                for j in range(0,k-1):
                    value=I[i-1][j]
                    if  self.cluster[i]!=self.cluster[value]:
                        count1+=1
                        I[i - 1][j]=0
        else:
            for i in range(1,self.n_items):
                if(i%100==0):
                    print (i)
                    print(f"count1avg={count1 / i}")
                for j in range(0,k-1):
                    value=I[i-1][j]
                    if self.sum[self.n_users + i][self.n_users + value] == 0 and self.cluster[i] != self.cluster[value]:
                        count1 += 1
                        I[i - 1][j] = 0

        return I
    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        # print(data_dict)
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        # print(data_dict)
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        self.diag = torch.from_numpy(diag).to(self.device)
        # print(diag)
        D = sp.diags(diag)
        L = D @ A @ D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        # print(L)
        # print(type(L))
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        # print(SparseL)
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight

        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers*2)):
            all_embeddings = torch.sparse.mm(self.norm_adj_mat, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers+1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def text_feat(self):
        model = SentenceTransformer('paraphrase-albert-small-v2')

        with open('dataset' + '/' + self.dataset + '/' + 'id-newid.json', 'r') as file:
            content = file.read()
        dic1 = json.loads(content)  # 将json格式文件转化为python的字典文件
        dic2 = {}
        with open('dataset' + '/' + self.dataset + '/' + self.dataset + '.item', 'r') as f:
            k = 0
            while True:
                k = k + 1
                if (k % 10000 == 0): print(k)
                a = f.readline()
                if a == '': break
                a = a.split('\t')
                id = a[0]
                if (id in dic1.keys()):
                    dic2[dic1[id]] = a[1]
        lst = []
        for i in range(1, len(dic2) + 1):
            lst.append(dic2[i])
            if (i % 100 == 0): print(i)
        lst = model.encode(lst)
        lst = np.array(lst)
        np.save('dataset' + '/' + self.dataset + '/' + 'text_feat.npy', lst)
        print("text_feat is ok")
    def image_feat(self):
        with open('dataset' + '/' + self.dataset + '/' + 'id-newid.json', 'r') as file:
            content = file.read()
        b1 = json.loads(content)  # 将json格式文件转化为python的字典文件
        with open('dataset' + '/' + self.dataset + '/' + 'newid-id.json', 'r') as file:
            content = file.read()
        b2 = json.loads(content)  # 将json格式文件转化为python的字典文件
        path = 'dataset' + '/'  + self.dataset + '.b'
        f = open(path, 'rb')
        k = 0
        d = dict()
        while True:
            if k % 10000 == 0:
                print(k)
            k = k + 1
            asin = f.read(10).decode('UTF-8')
            if asin == '': break
            a = array.array('f')
            a.fromfile(f, 4096)
            if asin in b1.keys():
                d[asin] = a.tolist()
        lst = []
        for i in range(1, len(b1) + 1):
            lst.append(d[b2[str(i)]])
        k = np.array(lst)
        np.save('dataset' + '/' + self.dataset + '/' + 'image_feat.npy', k)
        print("image_feat is ok")

    def knn_loss(self, node_embedding, knnlist,item,k):


        knnlist=torch.tensor(knnlist)
        _, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])
        item_embeddings = item_embeddings_all[item]
        item_embeddings = F.normalize(item_embeddings)
        item_knnlist=knnlist[item-1]
        total_loss=0
        for i in range(k-1):
            positem=item_embeddings_all[item_knnlist[:,i]]
            positem=F.normalize(positem)
            condition = torch.where(item_knnlist[:, i] > 0, 1, 0).cuda()

            pos_score_item = torch.mul(item_embeddings, positem).sum(dim=1)
            pos_score_item = torch.exp(pos_score_item / self.ssl_temp)

            ttl_score_item = torch.matmul(item_embeddings, item_embeddings.transpose(0, 1))
            ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

            knn_loss_item = -torch.log(pos_score_item / ttl_score_item)
            knn_loss_item = condition*knn_loss_item

            total_loss += self.ssl_weight * knn_loss_item.sum()
        return total_loss
    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None
        # batch里4096个交互数据,id 从1开始
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        # 全员的embedding
        user_all_embeddings, item_all_embeddings, embeddings_list= self.forward()
        init_embedding = embeddings_list[0]



        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        if self.image==1 and self.text==1:
            image_loss=self.knn_loss(init_embedding,self.imagelist, pos_item,self.imagek)
            text_loss=self.knn_loss(init_embedding,self.textlist, pos_item,self.textk)
            return mf_loss + self.reg_weight * reg_loss,self.image_weight*image_loss,self.text_weight*text_loss
        elif self.text==1:
            text_loss=self.knn_loss(init_embedding,self.textlist, pos_item,self.textk)
            return mf_loss + self.reg_weight * reg_loss,self.text_weight*text_loss
        elif self.image==1:
            image_loss=self.knn_loss(init_embedding,self.imagelist, pos_item,self.imagek)
            return mf_loss + self.reg_weight * reg_loss,self.image_weight*image_loss
        else:
            return mf_loss + self.reg_weight * reg_loss


    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list= self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
