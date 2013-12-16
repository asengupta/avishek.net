<?php
/*
Template Name: Index
*/
?>

<?php get_header(); ?>

		<!-- Maincontent start -->
		<div id="maincontent">
			<h2>Posts</h2>

<?php query_posts("showposts=10&paged=$paged"); while (have_posts()) : the_post(); ?>
			<div class="post">
				<h3><a href="<?php the_permalink() ?>" rel="bookmark" title="Permanent lenke til: <?php the_title_attribute(); ?>"><?php the_title(); ?></a></h3>
				<div class="date"><?php the_date(); ?> <?php the_time(); ?></div>
				<?php the_content(''); ?>
				<div class="postmeta">
					<span class="left"><?php if (function_exists('the_tags')) the_tags('Tags: '); ?></span>
					<span class="right"><?php comments_popup_link('No Comments', '1 Comment', '% Comments', 'comments'); ?></span>
				</div>
			</div>
<?php endwhile; ?>
<?php if (is_paged()) : ?>
			<div id="pagination" class="clear">
				<span class="left"><?php posts_nav_link('','&laquo; Back','') ?></span>
				<span class="right"><?php posts_nav_link('','','More &raquo;') ?></span>
			</div>
<?php endif; ?>

		</div>
		<!-- Maincontent end -->

<?php get_sidebar(); ?>

<?php get_footer(); ?>
