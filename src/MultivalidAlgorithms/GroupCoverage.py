import numpy as np
import jax
import jax.numpy as jnp
import optax
import functools

def group_coverage(tau, x_train, y_train, num_groups, group_fn=(lambda x: [True]), epochs=50, batch_size=2048):

    x_train, y_train = jnp.array(x_train, dtype=jnp.int32), jnp.array(y_train, dtype=jnp.float32)

    h = lambda x: jnp.where(jnp.array([jnp.equal(group_fn(x)[j], True) for j in range(num_groups)]))

    @functools.partial(jax.vmap, in_axes=(None, 0))
    @jax.jit
    def pred_fn_jax(theta, x):
        return jnp.dot(theta, jnp.array([(x % j == 0) for j in range(1, num_groups+1)], dtype=jnp.float32))

    @jax.jit
    def pinball_loss_jax(theta, x, y_true): 
        y_pred = pred_fn_jax(theta, x)
        return jnp.mean(jax.lax.max(tau*(y_true - y_pred), (1-tau)*(y_pred - y_true)))
    
    # for sgd: 0.1 lr, 0.5 momentum, nesterov=True, batch=2048, epochs=100

    optimizer = optax.adam(0.05)

    theta = jnp.zeros(num_groups)
    opt_state = optimizer.init(theta)

    @jax.jit
    def batch_update(x_batch, y_batch, theta, opt_state):
        loss, gradients = jax.value_and_grad(pinball_loss_jax)(theta, x_batch, y_batch)
        updates, opt_state = optimizer.update(gradients, opt_state)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state

    def TrainModelInBatches(x_train, y_train, epochs, theta, opt_state, batch_size=32):
        rng = np.random.RandomState(0)
        for i in range(1, epochs+1):
            inds = rng.permutation(len(x_train))
            x_train = x_train[inds]
            y_train = y_train[inds]
            batches = jnp.arange((x_train.shape[0]//batch_size)+1) ### Batch Indices

            # losses = [] ## Record loss of each batch
            for batch in batches:
                start = int(batch*batch_size)
                end = int(batch*batch_size+batch_size) if batch != batches[-1] else None
                x_batch, y_batch = x_train[start:end], y_train[start:end]

                theta, opt_state = batch_update(x_batch, y_batch, theta, opt_state)

                # losses.append(loss) ## Record Loss

            # print("Pinball Loss : {:.3f}".format(jnp.array(losses).mean()))

        return theta

    theta_opt = TrainModelInBatches(x_train, y_train, epochs=epochs, theta=theta, opt_state=opt_state, batch_size=batch_size)
    print("\nOptimal theta found:", theta_opt)

    return (lambda x: np.dot(np.array(theta_opt), group_fn(x))), np.array(theta_opt)
