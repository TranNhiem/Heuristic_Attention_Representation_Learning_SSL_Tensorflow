import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

######################################################################################
'''Supervised  Contrastive LOSS'''
######################################################################################


def multiclass_npair_loss(z, y):
    '''
    arg: z, hidden feature vectors(B_S[z], n_features)
    y: ground truth of shape (B_S[z])

    '''
    # Cosine similarity matrix
    z = tf.math.l2_normalize(z,  axis=1)
    Similarity = tf.matmul(z, z, transpose_b=True)
    loss = tfa.losses.npairs_loss(y, Similarity)
    return loss

# Supervised Contrastive Learning Paper


def multi_class_npair_loss_temperature(z, y, temperature):
    x_feature = tf.math.l2_normalize(z,  axis=1)
    similarity = tf.divide(
        tf.matmul(x_feature, tf.transpose(x_feature)), temperature)
    return tfa.losses.npairs_loss(y, similarity)


######################################################################################
'''Self-Supervised CONTRASTIVE LOSS'''
######################################################################################

'''N-Pair Loss'''


def multiclass_N_pair_loss(p, z):
    x_i = tf.math.l2_normalize(p, axis=1)
    x_j = tf.math.l2_normalize(z, axis=1)
    similarity = tf.matmul(x_i, x_j, transpose_b=True)
    batch_size = tf.shape(p)[0]
    contrastive_labels = tf.range(batch_size)

    # Simlarilarity treat as logic input for Cross Entropy Loss
    # Why we need the Symmetrized version Here??
    loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, similarity, from_logits=True)
    loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, tf.transpose(similarity), from_logits=True)

    return (loss_1_2+loss_2_1)/2


'''SimCLR Paper Nt-Xent Loss Keras Version'''
# Nt-Xent Loss Symmetrized


def nt_xent_symmetrize_keras(p, z, temperature):
    # cosine similarity the dot product of p,z two feature vectors
    x_i = tf.math.l2_normalize(p, axis=1)
    x_j = tf.math.l2_normalize(z, axis=1)
    similarity = (tf.matmul(x_i, x_j, transpose_b=True)/temperature)
    # the similarity from the same pair should be higher than other views
    batch_size = tf.shape(p)[0]  # Number Image within batch
    contrastive_labels = tf.range(batch_size)

    # Simlarilarity treat as logic input for Cross Entropy Loss
    # Why we need the Symmetrized version Here??
    loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, similarity, from_logits=True,)  # reduction=tf.keras.losses.Reduction.SUM
    loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
        contrastive_labels, tf.transpose(similarity), from_logits=True, )
    return (loss_1_2 + loss_2_1) / 2


'''SimCLR paper Asytemrize_loss V2'''

# Mask to remove the positive example from the rest of Negative Example


def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images
    # Ensure distinct pair of image get their similarity scores
    # passed as negative examples
    batch_size = batch_size.numpy()
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i+batch_size] = 0
    return tf.constant(negative_mask)


consie_sim_1d = tf.keras.losses.CosineSimilarity(
    axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(
    axis=2, reduction=tf.keras.losses.Reduction.NONE)


def nt_xent_asymetrize_loss_v1(p, z, temperature):  # negative_mask

    # L2 Norm
    batch_size = tf.shape(p)[0]
    batch_size = tf.cast(batch_size, tf.int32)
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    p_l2 = tf.math.l2_normalize(p, axis=1)
    z_l2 = tf.math.l2_normalize(z, axis=1)

    # Cosine Similarity distance loss

    # pos_loss = consie_sim_1d(p_l2, z_l2)
    pos_loss = tf.matmul(tf.expand_dims(p_l2, 1), tf.expand_dims(z_l2, 2))

    pos_loss = (tf.reshape(pos_loss, (batch_size, 1)))/temperature

    negatives = tf.concat([p_l2, z_l2], axis=0)
    # Mask out the positve mask from batch of Negative sample
    negative_mask = get_negative_mask(batch_size)

    loss = 0
    for positives in [p_l2, z_l2]:

        # negative_loss = cosine_sim_2d(positives, negatives)
        negative_loss = tf.tensordot(tf.expand_dims(
            positives, 1), tf.expand_dims(tf.transpose(negatives), 0), axes=2)
        l_labels = tf.zeros(batch_size, dtype=tf.int32)
        l_neg = tf.boolean_mask(negative_loss, negative_mask)

        l_neg = tf.reshape(l_neg, (batch_size, -1))
        l_neg /= temperature

        logits = tf.concat([pos_loss, l_neg], axis=1)  # [N, K+1]

        loss_ = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
        loss += loss_(y_pred=logits, y_true=l_labels)

    batch_size = tf.cast(batch_size, tf.float32)
    loss = loss/(2*batch_size)
    return loss


'''SimCLR Paper Nt-Xent Loss # SYMETRIC Loss'''
# Nt-Xent ---> N_Pair loss with Temperature scale
# Nt-Xent Loss (Remember in this case Concatenate Two Tensor Together)


def nt_xent_asymetrize_loss_v2(z,  temperature):
    '''The issue of design this loss two image is in one array
    when we multiply them that will lead two two same things mul together???

    '''
    # Feeding data (ALready stack two version Augmented Image)[2*bs, 128]
    z = tf.math.l2_normalize(z, axis=1)
    similarity_matrix = tf.matmul(
        z, z, transpose_b=True)  # pairwise similarity
    similarity = tf.exp(similarity_matrix / temperature)

    batch_size = tf.shape(z)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)

    ij_indices = tf.reshape(tf.range(z.shape[0]), shape=[-1, 2])
    ji_indices = tf.reverse(ij_indices, axis=[1])

    #[[0, 1], [1, 0], [2, 3], [3, 2], ...]
    positive_indices = tf.reshape(tf.concat(
        [ij_indices, ji_indices], axis=1), shape=[-1, 2])  # Indice positive pair

    # --> Output N-D array
    numerator = tf.gather_nd(similarity, positive_indices)
    # 2N-1 (sample)
    # mask that discards self-similarity
    negative_mask = 1 - tf.eye(z.shape[0])

    # compute sume across dimensions of Tensor (Axis is important in this case)
    # None sum all element scalar, 0 sum all the row, 1 sum all column -->1D metric
    denominators = tf.reduce_sum(
        tf.multiply(negative_mask, similarity), axis=1)
    losses = -tf.math.log(numerator/denominators)
    total_loss = tf.reduce_mean(losses)

    return total_loss, similarity, labels


def nt_xent_symetrize_loss_simcrl(hidden1, hidden2, LARGE_NUM,
                                  hidden_norm=True,
                                  temperature=1.0,
                                  ):
    """Compute loss for model.

    Args:
      hidden: hidden vector (`Tensor`) of shape (bsz, dim).
      hidden_norm: whether or not to use normalization on the hidden vector.
      temperature: a `floating` number for temperature scaling.

    Returns:
      A loss scalar.
      The logits for contrastive prediction task.
      The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden1 = tf.math.l2_normalize(hidden1, -1)  # 1
        hidden2 = tf.math.l2_normalize(hidden2, -1)
    #hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large,
                          transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large,
                          transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large,
                          transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large,
                          transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ab, logits_aa], 1))
    loss_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_ba, logits_bb], 1))
    loss = tf.reduce_mean(loss_a + loss_b) / 2

    return loss, logits_ab, labels


def nt_xent_symetrize_loss_object_level_whole_image_contrast(v1_object, v2_object, v1_background, v2_background,
                                                             image_rep1, image_rep2,
                                                             LARGE_NUM=1e-9, weight_loss=0.8, hidden_norm=True, temperature=1):
    """Compute loss for model.

    Args:
        Object Level Representation: 
        + v1_object, v2_object, v1_background, v2_background,
        Image Level Representation: 
        + image_rep1, image_rep2,

        hidden: hidden vector (`Tensor`) of shape (bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.

    Returns:
        A  Sumup of the loss scalar (Whole Image Represenation).
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
    """

    # ********* ----------------------- ***********
    # Contrastive Loss for Whole Image Representation
    # ********* ----------------------- ***********
    image_loss, whole_image_logits, lables_image = nt_xent_symetrize_loss_simcrl(image_rep1, image_rep2, LARGE_NUM,
                                                                                 hidden_norm=hidden_norm, temperature=temperature)

    # ********* ----------------------- ***********
    # Contrastive Loss for Whole Image Representation
    # ********* ----------------------- ***********
    object_rep_1 = tf.concat([v1_object, v1_background], axis=0)
    object_rep_2 = tf.concat([v2_object, v2_background], axis=0)
    object_loss, object_level_logits, lables_object_level = nt_xent_symetrize_loss_simcrl(object_rep_1, object_rep_2, LARGE_NUM,
                                                                                          hidden_norm=hidden_norm, temperature=temperature)
    total_loss = (weight_loss * object_loss + (1-weight_loss)*image_loss)/2

    # whole_image_logits, lables_image
    return total_loss, object_level_logits,  lables_object_level


def nt_xent_symetrize_loss_object_level_whole_image_contrast_v1(v1_object, v2_object, v1_background, v2_background,
                                                                image_rep1, image_rep2,
                                                                weight_loss=0.8,  temperature=1):
    """Compute loss for model.

    Args:
        Object Level Representation: 
        + v1_object, v2_object, v1_background, v2_background,
        Image Level Representation: 
        + image_rep1, image_rep2,

        hidden: hidden vector (`Tensor`) of shape (bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.

    Returns:
        A  Sumup of the loss scalar (Whole Image Represenation).
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
    """

    # ********* ----------------------- ***********
    # Contrastive Loss for Whole Image Representation
    # ********* ----------------------- ***********
    image_rep_1_2 = tf.concat([image_rep1, image_rep2], axis=0)
    image_loss, whole_image_logits, lables_image = nt_xent_asymetrize_loss_v2(image_rep_1_2,
                                                                              temperature=temperature)

    # ********* ----------------------- ***********
    # Contrastive Loss for Whole Image Representation
    # ********* ----------------------- ***********

    object_rep_1 = tf.concat([v1_object, v1_background], axis=0)
    object_rep_2 = tf.concat([v2_object, v2_background], axis=0)
    object_rep_1_2 = tf.concat([object_rep_1, object_rep_2], axis=0)
    object_loss, object_level_logits, lables_object_level = nt_xent_asymetrize_loss_v2(object_rep_1_2,
                                                                                       temperature=temperature)
    total_loss = (weight_loss * object_loss + (1-weight_loss)*image_loss)/2

    # ,whole_image_logits ,lables_image,
    return total_loss, object_level_logits,  lables_object_level


def binary_mask_nt_xent_object_backgroud_sum_loss(v1_object, v2_object, v1_background, v2_background,
                                                  LARGE_NUM=1e-9, alpha=0.8, temperature=1):
    '''
    Noted this Design 

    1. The contrasting 
        Similarity between object_1 and object_2, background_1 and background_2
        disimilarity object_1 <-> object_2
        disimilarity background_1 <-> background_2

        !! ATTENTION this design not Contrast object <--> background feature and vice versa

    2. Negative Pair will increasing 2 times 
        + The number of Lables will Increase to 2
        + the masks is the same of Original Contrast loss?

    3. Scaling Alpha value shound be (Mulitply -- Divided at the same)

    '''

    # L2 Norm
    batch_size = tf.shape(v1_object)[0]
    v1_object = tf.math.l2_normalize(v1_object, -1)
    v2_object = tf.math.l2_normalize(v2_object, -1)
    v1_background = tf.math.l2_normalize(v1_background, -1)
    v2_background = tf.math.l2_normalize(v2_background, -1)

    #INF = 1e9
    INF = LARGE_NUM

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)  # ??
    masks = tf.one_hot(tf.range(batch_size), batch_size)  # ??
    # ------------------------------------------------------
    # Similarity
    # ------------------------------------------------------
    # Object feature Simmilarity
    logits_o_aa = tf.matmul(v1_object, v1_object,
                            transpose_b=True) / temperature
    logits_o_aa = logits_o_aa - masks * INF  # remove the same samples
    logits_o_bb = tf.matmul(v2_object, v2_object,
                            transpose_b=True) / temperature
    logits_o_bb = logits_o_bb - masks * INF  # remove the same samples

    # Background Feature Simmilarity
    logits_b_aa = tf.matmul(v1_background, v1_background,
                            transpose_b=True) / temperature
    logits_b_aa = logits_b_aa - masks * INF
    logits_b_bb = tf.matmul(v2_background, v2_background,
                            transpose_b=True) / temperature
    logits_b_bb = logits_b_bb - masks * INF

    # ------------------------------------------------------
    # Disimilarity
    # ------------------------------------------------------
    # Object Disimilarity
    logits_o_ab = tf.matmul(v1_object, v2_object,
                            transpose_b=True) / temperature
    logits_o_ba = tf.matmul(v2_object, v1_object,
                            transpose_b=True) / temperature

    loss_o_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_o_ab, logits_o_aa], 1))
    loss_o_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_o_ba, logits_o_bb], 1))
    # Sum up all the Object loss together
    loss_object = tf.reduce_mean(loss_o_a + loss_o_b) / 2

    # Background Disimilarity
    logits_b_ab = tf.matmul(v1_background, v2_background,
                            transpose_b=True) / temperature
    logits_b_ba = tf.matmul(v2_background, v1_background,
                            transpose_b=True) / temperature

    loss_b_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_b_ab, logits_b_aa], 1))
    loss_b_b = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_b_ba, logits_b_bb], 1))
    # Sum up all the Object loss together
    loss_background = tf.reduce_mean(loss_b_a + loss_b_b) / 2

    total_loss = (alpha*loss_object + (1-alpha)*loss_background) / 2.0

    return total_loss, logits_o_ab,  labels  # logits_b_ab,


def binary_mask_nt_xent_object_backgroud_sum_loss_v1(object_f, background_f, alpha=0.8, temperature=1):
    # For Object
    z = object_f
    z = tf.math.l2_normalize(z, axis=1)
    ob_similarity_matrix = tf.matmul(
        z, z, transpose_b=True)  # pairwise similarity
    Ob_similarity = tf.exp(ob_similarity_matrix / temperature)

    # For Backgroud
    k = background_f
    k = tf.math.l2_normalize(k, axis=1)
    back_similarity_matrix = tf.matmul(
        k, k, transpose_b=True)  # pairwise similarity
    back_similarity = tf.exp(back_similarity_matrix / temperature)

    ij_indices = tf.reshape(tf.range(k.shape[0]), shape=[-1, 2])
    ji_indices = tf.reverse(ij_indices, axis=[1])
    #[[0, 1], [1, 0], [2, 3], [3, 2], ...]
    positive_indices = tf.reshape(tf.concat(
        [ij_indices, ji_indices], axis=1), shape=[-1, 2])  # Indice positive pair
    batch_size = tf.shape(z)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)

    # --> Output N-D array
    numerator_object = tf.gather_nd(Ob_similarity, positive_indices)
    numerator_back = tf.gather_nd(back_similarity, positive_indices)
    # 2N-1 (sample)
    # mask that discards self-similarity
    negative_mask = 1 - tf.eye(z.shape[0])

    # compute sume across dimensions of Tensor (Axis is important in this case)
    # None sum all element scalar, 0 sum all the row, 1 sum all column -->1D metric
    denominators_obj = tf.reduce_sum(
        tf.multiply(negative_mask, Ob_similarity), axis=1)
    losses_object = -tf.math.log(numerator_object/denominators_obj)

    denominators_back = tf.reduce_sum(
        tf.multiply(negative_mask, back_similarity), axis=1)
    losses_back = -tf.math.log(numerator_back/denominators_back)

    total_loss = tf.reduce_mean(
        alpha*losses_object + (1-alpha)*losses_object)/2

    return total_loss, Ob_similarity, labels  # back_similarity,


def binary_mask_nt_xent_only_Object_loss(v1_object, v2_object, LARGE_NUM, temperature=1):
    '''
    Noted Consideration Design 
    1. The contrasting Similarity between object_1 and object_2
    '''

    # L2 Norm
    batch_size = tf.shape(v1_object)[0]
    v1_object = tf.math.l2_normalize(v1_object, -1)
    v2_object = tf.math.l2_normalize(v2_object, -1)

    #INF = 1e9
    INF = LARGE_NUM

    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    # Object feature  dissimilar
    logits_o_aa = tf.matmul(v1_object, v1_object,
                            transpose_b=True) / temperature
    # print(logits_o_aa.shape)
    logits_o_aa = logits_o_aa - masks * INF  # remove the same samples
    logits_o_bb = tf.matmul(v2_object, v2_object,
                            transpose_b=True) / temperature
    logits_o_bb = logits_o_bb - masks * INF  # remove the same samples

    # Object feature  similar
    logits_o_ab = tf.matmul(v1_object, v2_object,
                            transpose_b=True) / temperature
    logits_o_ba = tf.matmul(v2_object, v1_object,
                            transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat(
        [logits_o_ab,  logits_o_aa], 1))

    loss_b = tf.nn.softmax_cross_entropy_with_logits(labels, tf.concat(
        [logits_o_ba, logits_o_bb], 1))

    loss = tf.reduce_mean(loss_a + loss_b) / 2.0

    return loss, logits_o_ab,  labels


######################################################################################
'''NON-CONTRASTIVE LOSS'''
####################################################################################

'''BYOL SYMETRIZE LOSS'''
# Symetric LOSS


def byol_symetrize_loss(p, z, temperature):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)
    # Calculate contrastive Loss
    batch_size = tf.shape(p)[0]
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    logits_ab = tf.matmul(p, z, transpose_b=True) / temperature
    # Measure similarity
    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    loss = 2 - 2 * tf.reduce_mean(similarities)
    return loss, logits_ab, labels


def symetrize_l2_loss_object_level_whole_image(o_1, o_2, b_1, b_2, img_1, img_2, weight_loss, temperature):
    """Compute loss for model.

    Args:
        Object Level Representation: 
        + v1_object, v2_object, v1_background, v2_background,
        Image Level Representation: 
        + image_rep1, image_rep2,

        hidden: hidden vector (`Tensor`) of shape (bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.

    Returns:
        A  Sumup of the loss scalar (Whole Image Represenation).
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
    """

    # ********* ----------------------- ***********
    # Contrastive Loss for Whole Image Representation
    # ********* ----------------------- ***********

    image_loss, whole_image_logits, lables_image = byol_symetrize_loss(img_1, img_2,temperature=temperature)

    # ********* ----------------------- ***********
    # Contrastive Loss for Whole Image Representation
    # ********* ----------------------- ***********

    object_rep_1 = tf.concat([o_1, b_1], axis=0)
    object_rep_2 = tf.concat([o_2, b_2], axis=0)

    object_loss, object_level_logits, lables_object_level = byol_symetrize_loss(object_rep_1, object_rep_2,
                                                                                temperature=temperature)
    total_loss = (weight_loss * object_loss + (1-weight_loss)*image_loss)/2

    # ,whole_image_logits ,lables_image,
    return total_loss, object_level_logits,  lables_object_level

def sum_symetrize_l2_loss_object_backg(o_1, o_2, b_1, b_2, alpha, temperature): 

    '''
    Noted this Design 

    1. The contrasting 
        Similarity between object_1 and object_2, background_1 and background_2
    3. Scaling Alpha value shound be for weighted loss between object and backgroud
    '''


    object_loss, object_logits, lables_object = byol_symetrize_loss(o_1, o_2,temperature=temperature)
    backg_loss, backg_logits, lables_back= byol_symetrize_loss(b_1, b_2,temperature=temperature)
    total_loss = (alpha * object_loss + (1-alpha)*backg_loss)/2
    # ,whole_image_logits ,lables_image,
    return total_loss, object_logits,  lables_object


'''Loss 2 SimSiam Model'''
# Asymetric LOSS


def simsam_loss(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))


def simsam_loss_non_stop_Gr(p, z):
    # The authors of SimSiam emphasize the impact of
    # the `stop_gradient` operator in the paper as it
    # has an important role in the overall optimization.
    #z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    # Negative cosine similarity (minimizing this is
    # equivalent to maximizing the similarity).
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))
