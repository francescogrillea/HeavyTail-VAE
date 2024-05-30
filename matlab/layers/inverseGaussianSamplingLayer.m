classdef inverseGaussianSamplingLayer < nnet.layer.Layer

    methods
        function layer = inverseGaussianSamplingLayer(args)
            % layer = normalSamplingLayer creates a sampling layer for VAEs.
            %
            % layer = samplingLayer(Name=name) also specifies the layer 
            % name.

            % Parse input arguments.
            arguments
                args.Name = "";
            end

            % Layer properties.
            layer.Name = args.Name;
            layer.Type = "InverseGaussianSampling";
            layer.Description = "Mean and shape inverse gaussian sampling";
            layer.OutputNames = ["out" "kl"];
            layer.NumOutputs = 2;
        end
    
        function [Z, KL] = predict(~,X)
            % [Z,KL] = predict(~,Z) Forwards input data through
            % the layer at prediction and training time and output the
            % result.
            %
            % Inputs:
            %         X - Concatenated input data where X(1:K,:) and 
            %             X(K+1:end,:) correspond to the mean and 
            %             shape, respectively, and K is the number 
            %             of latent channels.
            % Outputs:
            %         Z          - Sampled output
            %         KL         - KL divergence with IG(1,1)

            % Data dimensions.
            numLatentChannels = size(X,1)/2;
            miniBatchSize = size(X,2);

            % Split statistics.
            mu = X(1:numLatentChannels,:);
            shape = X(numLatentChannels+1:end,:);

            % Sample output.
            epsilon = randn(numLatentChannels,miniBatchSize,"like",X);
            y = epsilon.^2;
            
            Z = mu + (mu.^2).*y ./ (2*shape) - mu ./ (2*shape) .* sqrt(4*mu.*shape.*y + mu.^2 .* y.^2);
            
            mask = rand(numLatentChannels, miniBatchSize) <= (mu ./ (mu+Z));
            temp = (mu.^2) ./ Z;
            Z(~mask) = temp(~mask);

            % http://www.mathem.pub.ro/proc/bsgp-28/K28-ba-ZKH72.pdf
            KL = 0.5*(log(shape) + mu + 1./mu + 1./shape - 3);
        end

    end
    
end