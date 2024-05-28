function out = generate_samples(df, mu, sigma)
    x = df(mu(1), sigma(1));
    y = df(mu(2), sigma(2));
    out = [x,y];