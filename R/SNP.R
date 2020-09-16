rm(list = ls())
dev.new()
setwd("E:\\R语言程序")
raw <- read.delim("sheep_Eliminate outliers.raw",header=TRUE,sep=" ")
snp.cla <- read.delim("sheep_Eliminate outliers.map",header=FALSE,sep="\t")
colnames(snp.cla) <- c("cla","SNP","v1","v2")##c()把多个对象放到一起，组成向量
#fetch parts of data for testing 获取数据
choosed.class <- c("AD","HEA")
raw.test <- raw[raw$FID %in% choosed.class,-c(2:6)]
# $提取某个变量的结果 FID品种   %in%前面一个向量内的元素是否在后面一个向量中
#-c去除2-6列信息
names <- names(raw.test)


############################################################################################
#######################################补缺失值#############################################
#multiple imputation 
##partipate data into two group
tf <- function(x){
  return(any(is.na(x)))
  ##is.na()，返回值为逻辑值，TRUE代表缺失，否则为FALSE
}
exist.missing <- apply(raw.test,2,tf)
#apply循环 1：行 2：列
for.mice <- raw.test[,exist.missing]##缺失NA
complete <- raw.test[,!exist.missing]##不缺失
########
##mice
##library(mice)
##set.seed(1)
##miced <- complete(mice(for.mice,m=1,maxit = 1,method = 'lda'))
##combine 
##miced <- cbind(miced,complete)[,names]
#######
n.missing <- sum(is.na(for.mice))##缺失总数
sub <- sample(0:2,n.missing,replace=TRUE)
mx <- as.vector(for.mice)
##向量、矩阵转换
mx[is.na(mx)] <- sub
miced.before <- as.data.frame(mx)
##数据-数据框
miced <- cbind(miced.before,complete)[,names]
################################################################################################