classdef tStudentSamplingLayer < nnet.layer.Layer

    methods
        function layer = tStudentSamplingLayer(args)
            % layer = samplingLayer creates a sampling layer for VAEs.
            %
            % layer = samplingLayer(Name=name) also specifies the layer 
            % name.

            % Parse input arguments.
            arguments
                args.Name = "";
            end

            % Layer properties.
            layer.Name = args.Name;
            layer.Type = "TStudentSampling";
            layer.Description = "DoF t-Student sampling";
            layer.OutputNames = ["out" "DoF"];
        end
    
        function [Z, DoF] = predict(~,X)
            % [Z,DoF] = predict(~,Z) Forwards input data through
            % the layer at prediction and training time and output the
            % result.
            %
            % Inputs:
            %         X - Input data
            % Outputs:
            %         Z          - Sampled output
            %         DoF        - Degrees of Freedom.

            % Data dimensions
            numLatentChannels = size(X,1);
            miniBatchSize = size(X,2);

            DoF = X;

            % Sample output.
            z = randn(numLatentChannels, miniBatchSize, "like",X);

            % Chi2 sample
            chi = randn(numLatentChannels, numLatentChannels, miniBatchSize, "like",X);
            chi = sum(chi.^2, 1);
            
            Z = dlarray(trnd(extractdata(DoF)));
        end

    end
    
end