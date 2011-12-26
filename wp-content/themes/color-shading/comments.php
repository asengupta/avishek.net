<?php
/**
 * @package WordPress
 * @subpackage Default_Theme
 */

// Do not delete these lines
	if (!empty($_SERVER['SCRIPT_FILENAME']) && 'comments.php' == basename($_SERVER['SCRIPT_FILENAME']))
		die ('Please do not load this page directly. Thanks!');

	if ( post_password_required() ) { ?>
		<p class="nocomments">This post is password protected. Enter the password to view comments.</p>
	<?php
		return;
	}
?>

<!-- You can start editing here. -->

<?php if ( have_comments() ) : ?>
	<h3><?php comments_number(__('No Responses to','cover-wp'), __('One Response to','cover-wp'), __('% Responses to','cover-wp') ); ?> &quot;<?php the_title(); ?>&quot;</h3>

	<div class="commnav">
		<div class="alignleft"><?php previous_comments_link() ?></div>
		<div class="alignright"><?php next_comments_link() ?></div>
	</div>
    
	<ol class="commentlist">
    
    <?php if ( function_exists( 'wp_list_comments' ) )
	        wp_list_comments(array('style' => 'ol'));
	      else
		    echo 'Please update to WordPress verion 2.7 or later to enable comments'; ?>
    
	</ol>

	<div class="commnav">
		<div class="alignleft"><?php previous_comments_link() ?></div>
		<div class="alignright"><?php next_comments_link() ?></div>
	</div>
 <?php else : // this is displayed if there are no comments so far ?>

	<?php if ( comments_open() ) : ?>
		<!-- If comments are open, but there are no comments. -->

	 <?php else : // comments are closed ?>
		<!-- If comments are closed. -->
		<p class="nocomments"><?php _e('Comments are closed.','cover-wp') ?></p>

	<?php endif; ?>
<?php endif; ?>

<?php
$fields =  array(
	'author' => '<div class="form-row"><label for="name">' . ( $req ? '<q class="required">*</q>' : '' ) . ' Your Name</label><span>' . 
	            '<input type="text" name="author" id="author" value="' . esc_attr( $commenter['comment_author'] ) . '" tabindex="1" />' . '</span></div>',
	'email'  => '<div class="form-row"><label for="email">' . ( $req ? '<q class="required">*</q>' : '' ) . ' Your Email</label><span>' .
	            '<input type="text" name="email" id="email" value="' . esc_attr(  $commenter['comment_author_email'] ) . '" tabindex="2" />' . '</span></div>',
	'url'    => '<div class="form-row"><label for="url">Website</label><span>' .
	            '<input type="text"  name="url" id="url" value="' . esc_attr( $commenter['comment_author_url'] ) . '" size="22" tabindex="3" />' . '</span></div>',
	'url'    => '<div class="form-row"><label for="url">Website</label><span>' .
	            '<input type="text"  name="url" id="url" value="' . esc_attr( $commenter['comment_author_url'] ) . '" size="22" tabindex="3" />' . '</span></div>',
);

$defaults = array(
	'fields'               => apply_filters( 'comment_form_default_fields', $fields ),
	'comment_field'        => '<div class="form-row"><label for="content">' . ( $req ? '<q class="required">*</q>' : '' ) . ' Message</label><span><textarea name="content" cols="30" rows="20" tabindex="4"></textarea></span></div>',
	'comment_notes_before' => '',
	'comment_notes_after' => ''
);
?>

<?php if ( function_exists( 'comment_form' ) ) comment_form($defaults) ?>