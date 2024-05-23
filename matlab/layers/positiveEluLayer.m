classdef positiveEluLayer < nnet.layer.Layer

    methods
        function layer = positiveEluLayer(args)
            % layer = positiveEluLayer creates a shifted ELU activation layer.
            %
            % layer = samplingLayer(Name=name) also specifies the layer 
            % name.

            % Parse input arguments.
            arguments
                args.Name = "";
            end

            % Layer properties.
            layer.Name = args.Name;
            layer.Type = "EluLayer";
            layer.Description = "Shifted ELU activation layer";
            layer.OutputNames = "out";
            layer.NumOutputs = 1;
        end
    
        function Z = predict(~,X)
            % Z = predict(~,X) Forwards input data through elu activation
            %                  layer
            %
            % Inputs:
            %         X - Input Data
            % Outputs:
            %         Z - 1 + elu(x-1)
    
            Z = dlarray(zeros(size(X)));
            Z(X<0) = exp(X(X<0));
            Z(X>=0) = X(X>=0) + 1;
            
            % Z = (exp(X-1)).*(X<1) + X.*(X>=1); %  --- not working
        end

    end
    
end