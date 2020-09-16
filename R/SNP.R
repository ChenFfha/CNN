rm(list = ls())
dev.new()
setwd("E:\\R���Գ���")
raw <- read.delim("sheep_Eliminate outliers.raw",header=TRUE,sep=" ")
snp.cla <- read.delim("sheep_Eliminate outliers.map",header=FALSE,sep="\t")
colnames(snp.cla) <- c("cla","SNP","v1","v2")##c()�Ѷ������ŵ�һ���������
#fetch parts of data for testing ��ȡ����
choosed.class <- c("AD","HEA")
raw.test <- raw[raw$FID %in% choosed.class,-c(2:6)]
# $��ȡĳ�������Ľ�� FIDƷ��   %in%ǰ��һ�������ڵ�Ԫ���Ƿ��ں���һ��������
#-cȥ��2-6����Ϣ
names <- names(raw.test)


############################################################################################
#######################################��ȱʧֵ#############################################
#multiple imputation 
##partipate data into two group
tf <- function(x){
  return(any(is.na(x)))
  ##is.na()������ֵΪ�߼�ֵ��TRUE����ȱʧ������ΪFALSE
}
exist.missing <- apply(raw.test,2,tf)
#applyѭ�� 1���� 2����
for.mice <- raw.test[,exist.missing]##ȱʧNA
complete <- raw.test[,!exist.missing]##��ȱʧ
########
##mice
##library(mice)
##set.seed(1)
##miced <- complete(mice(for.mice,m=1,maxit = 1,method = 'lda'))
##combine 
##miced <- cbind(miced,complete)[,names]
#######
n.missing <- sum(is.na(for.mice))##ȱʧ����
sub <- sample(0:2,n.missing,replace=TRUE)
mx <- as.vector(for.mice)
##����������ת��
mx[is.na(mx)] <- sub
miced.before <- as.data.frame(mx)
##����-���ݿ�
miced <- cbind(miced.before,complete)[,names]
################################################################################################