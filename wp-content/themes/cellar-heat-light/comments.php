<?php // Do not delete these lines
	if ('comments.php' == basename($_SERVER['SCRIPT_FILENAME']))
		die ('Please do not load this page directly. Thanks!');

	if (!empty($post->post_password)) { // if there's a password
		if ($_COOKIE['wp-postpass_' . COOKIEHASH] != $post->post_password) {  // and it doesn't match the cookie
			?>

			<p class="nocomments">This post is password protected. Enter the password to view comments.</p>

			<?php
			return;
		}
	}

	/* This variable is for alternating comment background */
	$oddcomment = 'class="alt" ';
?>

<!-- You can start editing here. -->

<?php if ($comments) : ?>
	<div id="comment-list"><?php comments_number('No Comments', 'One comment', '% comments' );?> to...<br /><span class="bigger">&#8220;<?php the_title(); ?>&#8221;</span></div>

	<?php foreach ($comments as $comment) : ?>
		<div class="comment-left" <?php echo $oddcomment; ?>id="comment-<?php comment_ID() ?>">
			<div class="comment-meta">
            <img src="<?php gravatar("R", 40); ?>" alt="Avatar" class="avatar" /><br />
            <cite><?php comment_author_link() ?></cite>
			<?php if ($comment->comment_approved == '0') : ?>
			<em>Your comment is awaiting moderation.</em>
			<?php endif; ?>
			<br />
			<span class="comment-metadata">wrote on <?php comment_date('F jS, Y') ?> <?php edit_comment_link('edit','&nbsp;&nbsp;',''); ?></span>
            </div>
            <div class="comment-txt"><?php comment_text() ?></div>
            <br clear="all" />
		</div>

	<?php
		/* Changes every other comment to a different class */
		$oddcomment = ( empty( $oddcomment ) ) ? 'class="alt" ' : '';
	?>

	<?php endforeach; /* end for each comment */ ?>

	</ol>

 <?php else : // this is displayed if there are no comments so far ?>

	<?php if ('open' == $post->comment_status) : ?>
		<!-- If comments are open, but there are no comments. -->

	 <?php else : // comments are closed ?>
		<!-- If comments are closed. -->
		<p class="nocomments">Comments are closed.</p>

	<?php endif; ?>
<?php endif; ?>


<?php if ('open' == $post->comment_status) : ?>

<?php if ( get_option('comment_registration') && !$user_ID ) : ?>
<p>You must be <a href="<?php echo get_option('siteurl'); ?>/wp-login.php?redirect_to=<?php the_permalink(); ?>">logged in</a> to post a comment.</p>
<?php else : ?>

<form action="<?php echo get_option('siteurl'); ?>/wp-comments-post.php" method="post" id="commentform">

<?php if ( $user_ID ) : ?>

<p>Logged in as <a href="<?php echo get_option('siteurl'); ?>/wp-admin/profile.php"><?php echo $user_identity; ?></a>. <a href="<?php echo get_option('siteurl'); ?>/wp-login.php?action=logout" title="Log out of this account">Logout &raquo;</a></p>

<?php else : ?>
<div class="comments-PII">
<p><label for="author">Name:</label><br />
<input type="text" name="author" id="author" value="<?php echo $comment_author; ?>" tabindex="1" /><br />
<span class="note"><?php if ($req) echo "required"; ?></span></p>

<p><label for="email">Email Address:</label><br />
<input type="text" name="email" id="email" value="<?php echo $comment_author_email; ?>" tabindex="2" /><br />
<span class="note"><?php if ($req) echo "required -"; ?> won't be displayed</span></p>

<p><label for="url">Website URL:</label><br />
<input type="text" name="url" id="url" value="<?php echo $comment_author_url; ?>" tabindex="3" /></p>
</div>
<?php endif; ?>

<!--<p><small><strong>XHTML:</strong> You can use these tags: <code><?php echo allowed_tags(); ?></code></small></p>-->
<div class="comments-PII-2">
<p>Your Comment:<br />
<textarea name="comment" id="comment" tabindex="4"></textarea></p>

<p><input name="submit" type="submit" id="submit" tabindex="5" value="Add Comment &raquo;" />
<input type="hidden" name="comment_post_ID" value="<?php echo $id; ?>" />
</p>
</div>
<?php do_action('comment_form', $post->ID); ?>

</form>

<?php endif; // If registration required and not logged in ?>

<?php endif; // if you delete this the sky will fall on your head ?>
