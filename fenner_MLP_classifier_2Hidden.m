function [Error_rate]=fenner_MLP_classifier_2Hidden(data, class, cW0, cW1, cW2)
%Assume
[n,m]=size(data);
[K,t]=size(cW0);
[J,i]=size(cW1);
[L,j]=size(cW2);
errorSum=0;

for colData=1:1:m
        
   %formats data for easy calulations.
        tempData=ones(n+1,1);
        for i=1:1:n
            tempData(i,1)=data(i,colData);   
        end

        %This holds all the sums of inputs filtered through tanh().
        %Initialize.
        Hmatrix=tanh(cW0*tempData);

        %Creates the Hidden-layer-1's perceptron values, plus a bias. 
        tempH=ones(K+1,1);
        for row=1:1:K
            tempH(row,1)=Hmatrix(row,1);  
        end
        
        Zmatrix=tanh(cW1*tempH);
        
        
        %Creates the Hidden-layer-2's perceptron values, plus a bias. 
        tempZ=ones(J+1,1);
        for row=1:1:J
            tempZ(row,1)=Zmatrix(row,1);  
        end
       
        %Gets the outputs.
        Ymatrix=tanh(cW2*tempZ);
    
    %Max select
    [M,I] = max(Ymatrix);
    
    if (I-1)==class(colData,1)
        errorSum=errorSum+1;
    end

end

Error_rate=((errorSum/m)*100);
end

