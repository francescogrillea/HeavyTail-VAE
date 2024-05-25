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
            layer.NumOutputs = 2;
        end
    
        function [Z, DoF] = predict(this, X)
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

            % Normal sample
            z = randn(numLatentChannels, miniBatchSize, "like",X);

            % Chi2 sample
            %   Chi2(nu) = Gamma(nu/2, 2)
            c = this.gammarand(X/2, 2*ones(size(X)));

            % T sample
            %   T(nu) = N(0,1) * sqrt(nu / Chi2(nu))
            Z = z .* sqrt(X ./ c);
        end

        function Z = gammarand(~, k, theta)
            % Generate random samples from the gamma distribution with
            % shape k and scale theta
            % (see https://en.wikipedia.org/wiki/Gamma_distribution#Random_variate_generation)
            %
            % Syntax
            %   Z = gammarand(k, theta)
            %
            %  Input Arguments
            %    k - Shape parameter
            %      matrix of nonnegative scalar values
            %    theta - Scale parameter
            %      matrix of nonnegative scalar values
            %
            %  Output Arguments
            %    Z - Gamma random numbers
            %      matrix of nonnegative scalar values
            %      (with the same shape of a and b)

            if nargin < 3
                error("Must provide shape and scale")
            end

            num_el = numel(k);

            % Generate Gamma(shape+1, 1) so that shape+1 >= 1
            d = (k+1) - 1/3; d = reshape(d, 1, []);
            c = 1 ./ (3 * sqrt(d)); c = reshape(c, 1, []);
            
            Z = zeros([1, num_el], "like", d);
            ok = false([1, num_el]);

            while ~all(ok)
                not_ok = sum(~ok, "all");

                X = randn([1, not_ok]);
                v = (1 + c(~ok).*X).^3;
                Z(~ok) = d(~ok).*v;

                U = rand([1, not_ok]);
                ok(~ok) = (v > 0) & (log(U) < (X.^2)/2 + d(~ok).*(1-v+log(v)));
            end

            % Convert to Gamma(shape, 1)
            Z = Z .* rand([1, num_el]).^(1./reshape(k,1,[]));

            % Rescale to Gamma(shape, scale)
            Z = theta .* reshape(Z, size(k));
        end
    end
    
end