<?php // Do not delete these lines
	if ('comments.php' == basename($_SERVER['SCRIPT_FILENAME']))
		die ('Please do not load this page directly. Thanks!');

		/* This variable is for alternating comment background */
		$oddcomment = 'alt';
?>

<!-- You can start editing here. -->

<?php if ($comments) : ?>

<h2>Comments</h2>

	<?php $commentnumber = 1?>
	<?php foreach ($comments as $comment) : ?>

		<div id="comment-<?php comment_ID() ?>" class="comment <?php echo $oddcomment; if ($comment->comment_author_email == get_the_author_email()) { echo ' authorcomment'; } ?>">
			<div class="commentavatar">
				<?php echo get_avatar( $comment, 64 ); ?>
			</div>
			<div class="commenttext">
				<h4><?php comment_author_link() ?></h4>
				<div class="date"><?php comment_date(); ?> <?php the_time(); ?></div>
				<?php if ($comment->comment_approved == '0') : ?>
				<em>Your comment is awaiting moderation</em>
				<?php endif; ?>
				<?php comment_text() ?>
			</div>
		</div>

	<?php /* Changes every other comment to a different class */
		if ('alt' == $oddcomment) $oddcomment = '';
		else $oddcomment = 'alt';
	?>

	<?php endforeach; /* end for each comment */ ?>

		<div class="commentmeta">
<?php if ($post->ping_status == "open") { ?>
			<span class="left"><a href="<?php trackback_url(display); ?>">Trackback URI</a></span>
<?php }; ?>
<?php if ($post-> comment_status == "open") { ?>
			<span class="right"><?php comments_rss_link('Comments RSS'); ?></span>
<?php }; ?>
		</div>

<?php else : // this is displayed if there are no comments so far ?>

  <?php if ('open' == $post-> comment_status) : ?>
		<!-- If comments are open, but there are no comments. -->

	 <?php else : // comments are closed ?>
		<!-- If comments are closed. -->
		<p class="nocomments">Comments are closed at this time.</p>

	<?php endif; ?>
<?php endif; ?>

<?php if ('open' == $post-> comment_status) : ?>

<h2 id="respond">Leave Comment</h2>

<?php if ( get_option('comment_registration') && !$user_ID ) : ?>
<p>You must be <a href="<?php echo get_option('siteurl'); ?>/wp-login.php?redirect_to=<?php the_permalink(); ?>">logged in</a> to post a comment.</p>
<?php else : ?>

<form action="<?php echo get_option('siteurl'); ?>/wp-comments-post.php" method="post" id="commentform">

<?php if ( $user_ID ) : ?>

<p>Logged in as <a href="<?php echo get_option('siteurl'); ?>/wp-admin/profile.php"><?php echo $user_identity; ?></a>. <a href="<?php echo get_option('siteurl'); ?>/wp-login.php?action=logout" title="<?php _e('Log out of this account') ?>">Logout &raquo;</a></p>

<?php else : ?>

<p><input type="text" class="textbox" name="author" id="author" value="<?php echo $comment_author; ?>" size="30" tabindex="1" />
<label for="author">Name <?php if ($req) _e('(required)'); ?></label></p>

<p><input type="text" class="textbox" name="email" id="email" value="<?php echo $comment_author_email; ?>" size="30" tabindex="2" />
<label for="email">Mail (hidden) <?php if ($req) _e('(required)'); ?></label></p>

<p><input type="text" class="textbox" name="url" id="url" value="<?php echo $comment_author_url; ?>" size="30" tabindex="3" />
<label for="url">Website</label></p>

<?php endif; ?>

<!--<p><small><strong>XHTML:</strong> You can use these tags: <?php echo allowed_tags(); ?></small></p>-->

<p><textarea name="comment" id="comment" cols="100%" rows="6" tabindex="4"></textarea></p>

<p>
  <input name="submit" type="submit" id="submit" tabindex="5" value="Submit Comment" />
<input type="hidden" name="comment_post_ID" value="<?php echo $id; ?>" />
</p>
<?php do_action('comment_form', $post->ID); ?>

</form>

<?php endif; // If registration required and not logged in ?>
<?php endif; // if you delete this the sky will fall on your head ?>