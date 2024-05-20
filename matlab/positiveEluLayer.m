classdef positiveEluLayer < nnet.layer.Layer

    methods
        function layer = normalSamplingLayer(args)
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
            layer.Type = "NormalSampling";
            layer.Description = "Mean and log-variance normal sampling";
            layer.OutputNames = ["out" "mean" "log-variance"];
        end
    
        function Z = predict(~,X)
            % Z = predict(~,X) Forwards input data through elu activation
            %                  layer
            %
            % Inputs:
            %         X - Input Data
            % Outputs:
            %         Z - 1 + elu(x-1)
    
            Z = exp(X).*(X<0) + (X+1).*(X>=0);
            % Z = (exp(X-1)).*(X<1) + X.*(X>=1)   --- not working
        end

    end
    
end