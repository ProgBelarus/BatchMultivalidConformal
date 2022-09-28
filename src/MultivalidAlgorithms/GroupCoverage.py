import numpy as np
import jax
import jax.numpy as jnp
import functools
import optax

def group_coverage(tau, x_train, y_train, num_groups, group_fn=(lambda x: [True]), \
                   batch_size=2048, lr=0.05, epochs=50, \
                   opt_alg='sgd', sgd_momentum=0.5, sgd_nesterov=True, \
                   early_stop=True, early_stop_epochs=10000, eps=0.001, rounds_without_improv=50, valid_size=0.2):

    if early_stop:
        from sklearn.model_selection import train_test_split
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=valid_size)
        x_valid, y_valid = jnp.array(x_valid, dtype=jnp.int32), jnp.array(y_valid, dtype=jnp.float32)
        epochs = early_stop_epochs

    x_train, y_train = jnp.array(x_train, dtype=jnp.int32), jnp.array(y_train, dtype=jnp.float32)

    @functools.partial(jax.vmap, in_axes=(None, 0))
    @jax.jit
    def pred_fn_jax(theta, x):
        return jnp.dot(theta, jnp.array(group_fn(x), dtype=jnp.float32))

    @jax.jit
    def pinball_loss_jax(theta, x, y_true): 
        y_pred = pred_fn_jax(theta, x)
        return jnp.mean(jax.lax.max(tau*(y_true - y_pred), (1-tau)*(y_pred - y_true)))
    
    if opt_alg == 'sgd':
        optimizer = optax.sgd(learning_rate=lr, momentum=sgd_momentum, nesterov=sgd_nesterov)
    elif opt_alg == 'adam':
        optimizer = optax.adam(lr)
    else:
        optimizer = optax.sgd(learning_rate=0.05, momentum=0.5, nesterov=True)

    theta = jnp.zeros(num_groups)
    opt_state = optimizer.init(theta)

    @jax.jit
    def batch_update(x_batch, y_batch, theta, opt_state):
        loss, gradients = jax.value_and_grad(pinball_loss_jax)(theta, x_batch, y_batch)
        updates, opt_state = optimizer.update(gradients, opt_state)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state

    def training_loop(x_train, y_train, x_valid, y_valid, epochs, theta, opt_state, batch_size):
        rng = np.random.RandomState(0)

        if early_stop:
            best_valid_loss = jnp.inf
            rounds_without_improvement = 0
            valid_loss_prev = jnp.inf
        
        for i in range(1, epochs+1):
            inds = rng.permutation(len(x_train))
            x_train = x_train[inds]
            y_train = y_train[inds]
            batches = jnp.arange((x_train.shape[0]//batch_size)+1) # batch indices

            for batch in batches:
                start = int(batch*batch_size)
                end = int(batch*batch_size+batch_size) if batch != batches[-1] else None
                x_batch, y_batch = x_train[start:end], y_train[start:end]

                theta, opt_state = batch_update(x_batch, y_batch, theta, opt_state)

            if early_stop:
                valid_loss_curr = pinball_loss_jax(theta, x_valid, y_valid)

                rounds_without_improvement += 1
                
                if valid_loss_curr < best_valid_loss - eps:
                    rounds_without_improvement = 0
                if rounds_without_improvement >= rounds_without_improv:
                    break
                
                if valid_loss_curr < best_valid_loss:
                    best_valid_loss = valid_loss_curr
                valid_loss_prev = valid_loss_curr
                
        return theta

    theta_opt = training_loop(x_train, y_train, x_valid, y_valid, epochs, theta, opt_state, batch_size)
    print("\nOptimal theta found:", theta_opt)

    return (lambda x: np.dot(np.array(theta_opt), group_fn(x))), np.array(theta_opt)
