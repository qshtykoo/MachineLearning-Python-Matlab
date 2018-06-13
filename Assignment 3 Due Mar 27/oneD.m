Rn = 0;
count = 1;
z_m = 0;

    p1 = rand;
    p2 = 1-p1;
    P_t = [p1 p2];

while Rn <= log(2)
    z1 = rand*randi(10);
    z2 = rand*randi(10);
    Z_t = [z1 z2]';
    if z1<=z2
       z_m = z1;
    else
       z_m = z2;
    end
    sum_lm = -log(P_t*Z_t);
    Rn = sum_lm - z_m;
    count = count + 1;
    if count == 500 
        break;
    end
end