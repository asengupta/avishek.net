<?php

// Content Width

if(!isset($content_width)) $content_width = 591;

// Automatic Feed Links

add_theme_support('automatic-feed-links');

// Custom Background

add_custom_background();

// Display Page Title

function display_title($s)
{
	if (is_404())
		_e('Page Not Found ','cover-wp');
	elseif (is_search())
	{
		_e('Search Results for ','cover-wp');
		echo '&quot;'.$s.'&quot; ';
	}
	elseif (is_tag())
	{
		_e('Entries tagged with ','cover-wp');
		echo '&quot;'.single_tag_title("", false).'&quot; ';
	}
	else wp_title(' ');
	if(wp_title(' ', false)) echo ' | ';
	bloginfo('name');
}

// Widgetable Area

if ( function_exists('register_sidebar') ) 
{     
  register_sidebar(array('name' => 'Sidebar','before_widget' => '<div id="%1$s" class="widget %2$s">','after_widget' => '</div><!-- /widget -->','before_title' => '<h3>','after_title' => '</h3>'));
}

// Excerpt Read More

function new_excerpt_more($more) {
	return '&#8230;';
}
add_filter('excerpt_more', 'new_excerpt_more');

// Recent Comments

function latest_comments()
{
  global $wpdb;

  $comments = get_comments('status=approve&number=7');
  if ($comments)
  {
    foreach ($comments as $comment)
    {
      echo '<li class="recentcomments">';
      echo '<a href="'.get_permalink($comment->comment_post_ID).'#comment-'.$comment->comment_ID.'">'.$comment->comment_author.'</a>';
      echo ' '.__('on','cover-wp').' <a href="'.get_permalink($comment->comment_post_ID).'">'.get_the_title($comment->comment_post_ID).'</a><p><span>';
	  $content = $comment->comment_content;
	  $content = substr_replace($content, '', 80);
      echo '<em>&quot;'.strip_tags ($content).'&#8230;&quot;</em></span></p></li>';
    }
  }
}

// Unregister WP-Page-Navi Stylesheet

add_action( 'wp_print_styles', 'my_deregister_styles', 100 );

function my_deregister_styles() {
	wp_deregister_style( 'wp-pagenavi' );
}

// Custom Color Scheme

require_once ( get_template_directory() . '/color-scheme.php' );

add_action('wp_head', 'color_scheme', 9);

/* *** WordPress 3.0 Features *** */

// Custom Menus

function mytheme_addmenus() {
	if ( function_exists( 'register_nav_menus' ) )
		register_nav_menus(
		array(
			'main_nav' => 'Top Navigation', // You can add more menus here
		)
	);
}

add_action( 'init', 'mytheme_addmenus' );

function mytheme_nav() {
    if ( function_exists( 'wp_nav_menu' ) )
        wp_nav_menu( array('theme_location' => 'main_nav', 'container' => '', 'menu_id' => 'top-nav', 'menu_class' => 'clearfix', 'fallback_cb' => 'mytheme_nav_fallback') );
    else
        mytheme_nav_fallback();
}

function mytheme_nav_fallback() { ?>
    <ul id="top-nav" class="clearfix">
      <li<?php if(is_home()) echo ' class="current_page_item"'; ?>><a href="<?php echo home_url(); ?>/">Home</a></li>
	  <?php wp_list_pages('sort_column=menu_order&title_li='); ?>
    </ul><?php
}

add_editor_style();

?>